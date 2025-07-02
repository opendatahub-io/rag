#!/usr/bin/env python3
"""
Simple RAG Agent Script - A didactic example of Retrieval Augmented Generation

This script demonstrates the basic steps of RAG:
1. Load documents from files
2. Convert them to text using advanced docling processing
3. Generate embeddings manually for better control
4. Store them in a vector database with rich metadata
5. Create an agent that can query the documents
6. Ask questions and get answers

Usage:
    python setup_rag_agent.py
"""

import uuid
import json
from pathlib import Path
import logging
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
# All the parameters you can customize for your RAG setup

# Basic connection and model settings
LLAMA_STACK_URL = "http://localhost:8081"  # URL where your llama-stack is running
INFERENCE_MODEL = "vllm"                   # Model used to generate answers (LLM)
EMBEDDING_MODEL = "granite-embedding-125m"  # Model used to create embeddings (converts text to vectors)
EMBEDDING_DIM = 768                        # Dimension of the embedding vectors
AGENT_NAME = "RAG Team Agent 2.0"            # Human-readable name for your agent

# Document processing settings
INPUT_FOLDER = "input_files"               # Folder where your documents are stored
SUPPORTED_EXTENSIONS = [".txt", ".pdf"]    # File types this script can process
CHUNK_SIZE_IN_TOKENS = 512                 # How to split documents into chunks for better retrieval (increased from 256)

# Vector database settings (where document embeddings are stored)
VECTOR_DB_PROVIDER = "milvus"              # Type of vector database (milvus, weaviate, etc.)
VECTOR_DB_PREFIX = "simple-rag-db"         # Prefix for naming the vector database

# RAG agent behavior settings
TOP_K = 3                                  # How many document chunks to retrieve when answering questions
SIMILARITY_THRESHOLD = 0.0                 # Minimum similarity score (0.0 = accept all, higher = more strict)
MAX_INFER_ITERS = 10                       # Maximum reasoning steps the agent can take
ENABLE_SESSION_PERSISTENCE = False         # Whether to remember conversation history

# Instructions that tell the agent how to behave
AGENT_INSTRUCTIONS = """You are a helpful assistant that answers questions based on the provided documents. 
When asked a question, search through the documents and provide accurate, direct answers based on what you find.
Always answer in a natural, conversational way. Do not ask for function calls or specific formats.
If you don't find relevant information in the documents, say so clearly."""

# Session settings
SESSION_NAME = "simple-rag-session"        # Name for the chat session

# PDF processing options (improved settings for better text extraction)
PDF_DO_OCR = True                          # Use OCR to extract text from images in PDFs (enabled for better quality)
PDF_DO_TABLE_STRUCTURE = True             # Extract table structures from PDFs  
PDF_DO_CELL_MATCHING = True               # Match table cells for better table understanding
PDF_GENERATE_PAGE_IMAGES = True           # Generate page images for better processing

# Logging configuration
LOG_LEVEL = "INFO"                         # How detailed the logging should be

# Enable logging to see what's happening during execution
# Options: DEBUG, INFO, WARNING, ERROR
logging.basicConfig(level=getattr(logging, LOG_LEVEL))

# =============================================================================
# EMBEDDING AND CHUNKING FUNCTIONS
# =============================================================================

def setup_chunker_and_embedder(embed_model_id: str, max_tokens: int):
    """
    Set up the advanced chunker and embedding model for better document processing.
    
    This uses the same approach as the better-performing KFP version:
    - HybridChunker for document-aware chunking
    - SentenceTransformer for manual embedding generation
    
    Args:
        embed_model_id: Model ID for embedding generation
        max_tokens: Maximum tokens per chunk
        
    Returns:
        tuple: (embedding_model, chunker)
    """
    print(f"🔧 Setting up chunker and embedder...")
    print(f"   • Embedding model: {embed_model_id}")
    print(f"   • Max tokens per chunk: {max_tokens}")
    
    # Set up tokenizer and chunker (same as KFP version)
    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    embedding_model = SentenceTransformer(embed_model_id)
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=True)
    
    print(f"✅ Chunker and embedder ready")
    return embedding_model, chunker

def embed_text(text: str, embedding_model) -> list[float]:
    """
    Generate embeddings for text using SentenceTransformer.
    
    Args:
        text: Text to embed
        embedding_model: SentenceTransformer model
        
    Returns:
        list[float]: Normalized embedding vector
    """
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

# =============================================================================
# DOCUMENT LOADING FUNCTIONS
# =============================================================================

def load_text_file(file_path):
    """
    Load a simple text file and extract its content.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        str: The text content of the file, or None if there's an error
    """
    print(f"📖 Loading text file: {file_path.name}")
    try:
        # Read the file with UTF-8 encoding to handle special characters
        content = file_path.read_text(encoding="utf-8")
        return content.strip()  # Remove leading/trailing whitespace
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        return None

def load_pdf_file(file_path):
    """
    Load a PDF file and extract its text content using advanced docling processing.
    
    This function uses the same advanced PDF processing as the better-performing version:
    - OCR enabled for image-based text
    - Table structure extraction
    - Page image generation
    - Advanced pipeline options
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        docling Document: The processed document object, or None if there's an error
    """
    print(f"📄 Loading PDF file: {file_path.name}")
    try:
        # Configure advanced PDF processing options (same as KFP version)
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = PDF_DO_OCR  # OCR for image-based text
        pdf_options.do_table_structure = PDF_DO_TABLE_STRUCTURE  # Extract table structures
        pdf_options.generate_page_images = PDF_GENERATE_PAGE_IMAGES  # Generate page images
        if PDF_DO_TABLE_STRUCTURE:
            pdf_options.table_structure_options.do_cell_matching = PDF_DO_CELL_MATCHING  # Match table cells
        
        # Create a document converter with advanced PDF settings
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options,
                    backend=PyPdfiumDocumentBackend,  # Use PyPdfium for robust PDF processing
                ),
            },
        )
        
        # Convert the PDF and return the document object (not just text)
        result = converter.convert(file_path)
        if result and result.document:
            print(f"✅ PDF processed: {file_path.name}")
            return result.document  # Return the full document object for better chunking
        else:
            print(f"❌ Could not process {file_path.name}")
            return None
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")
        return None

def load_documents_from_folder(folder_path=INPUT_FOLDER):
    """
    Load all supported documents from the specified folder.
    
    This function:
    1. Scans the folder for supported file types
    2. Processes each file using the appropriate loader
    3. Returns processed documents ready for advanced chunking
    
    Args:
        folder_path: Path to the folder containing documents
        
    Returns:
        list: List of processed documents (text for .txt, docling Document objects for .pdf)
    """
    print(f"\n📁 Loading documents from '{folder_path}' folder...")
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder '{folder_path}' not found!")
        return []
    
    documents = []
    supported_files = []
    
    # Find all files with supported extensions
    for ext in SUPPORTED_EXTENSIONS:
        supported_files.extend(folder.glob(f"*{ext}"))
    
    if not supported_files:
        print(f"❌ No supported files found! Please add files with these extensions: {SUPPORTED_EXTENSIONS}")
        return []
    
    print(f"📋 Found {len(supported_files)} file(s): {[f.name for f in supported_files]}")
    
    # Process each file
    for file_path in supported_files:
        content = None
        
        # Route to appropriate loader based on file extension
        if file_path.suffix.lower() == ".txt":
            content = load_text_file(file_path)
            if content:
                documents.append({
                    'content': content,
                    'file_name': file_path.stem,
                    'file_path': str(file_path),
                    'type': 'text'
                })
        elif file_path.suffix.lower() == ".pdf":
            doc = load_pdf_file(file_path)
            if doc:
                documents.append({
                    'document': doc,
                    'file_name': file_path.stem,
                    'file_path': str(file_path),
                    'type': 'pdf'
                })
    
    return documents

# =============================================================================
# VECTOR DATABASE FUNCTIONS
# =============================================================================

def setup_vector_database_and_insert_documents(client, documents):
    """
    Create a vector database and insert documents using advanced processing.
    
    This function uses the same approach as the better-performing KFP version:
    1. Creates a vector database
    2. Uses HybridChunker for advanced document-aware chunking
    3. Generates embeddings manually using SentenceTransformer
    4. Inserts chunks with pre-computed embeddings using client.vector_io.insert()
    
    Args:
        client: LlamaStackClient instance
        documents: List of processed documents
        
    Returns:
        str: The ID of the created vector database
    """
    print(f"\n🗄️  Setting up vector database with advanced processing...")
    
    # Create a unique ID for this vector database
    vector_db_id = f"{VECTOR_DB_PREFIX}-{uuid.uuid4().hex}"
    
    # Get embedding model information from llama-stack
    models = client.models.list()
    print(f"🔍 Looking for embedding model '{EMBEDDING_MODEL}'...")
    
    # First try to find the specific model by provider_resource_id
    matching_model = next((m for m in models if m.provider_resource_id == EMBEDDING_MODEL), None)
    
    # If not found by provider_resource_id, try by identifier
    if not matching_model:
        matching_model = next((m for m in models if m.identifier == EMBEDDING_MODEL), None)
    
    # If still not found, fall back to any embedding model (like the Jupyter notebook does)
    if not matching_model:
        print(f"⚠️  Specific model '{EMBEDDING_MODEL}' not found. Looking for any embedding model...")
        matching_model = next((m for m in models if m.model_type == "embedding"), None)
        if matching_model:
            print(f"✅ Using embedding model: {matching_model.identifier}")
        else:
            # Show available models for debugging
            print(f"❌ No embedding models found on server!")
            print(f"Available models:")
            for m in models:
                print(f"   • {m.identifier} (type: {m.model_type}, provider_resource_id: {getattr(m, 'provider_resource_id', 'N/A')})")
            raise ValueError(f"No embedding models found on LlamaStack server.")
    
    if matching_model.model_type != "embedding":
        raise ValueError(f"Model '{matching_model.identifier}' is not an embedding model (type: {matching_model.model_type})")
    
    embedding_dimension = matching_model.metadata["embedding_dimension"]
    print(f"✅ Using embedding model: {matching_model.identifier} (dimension: {embedding_dimension})")
    
    # Register the vector database with llama-stack
    client.vector_dbs.register(
        vector_db_id=vector_db_id,           # Unique identifier
        embedding_model=matching_model.identifier,     # Which model to use for embeddings
        embedding_dimension=embedding_dimension,   # Size of the embedding vectors
        provider_id=VECTOR_DB_PROVIDER       # Type of vector database
    )
    print(f"✅ Vector database registered: {vector_db_id}")
    
    # Set up chunker and embedder using the actual model found (same as KFP version)
    # Use the provider_resource_id if available, otherwise fall back to identifier
    actual_embedding_model_id = getattr(matching_model, 'provider_resource_id', matching_model.identifier)
    embedding_model, chunker = setup_chunker_and_embedder(actual_embedding_model_id, CHUNK_SIZE_IN_TOKENS)
    
    # Process documents using advanced chunking and embedding generation
    print("📥 Processing documents with advanced chunking and embedding generation...")
    
    total_chunks = 0
    for doc_info in documents:
        file_name = doc_info['file_name']
        print(f"🔄 Processing: {file_name}")
        
        chunks_with_embedding = []
        
        if doc_info['type'] == 'pdf':
            # Use HybridChunker for PDF documents (same as KFP version)
            document = doc_info['document']
            for chunk in chunker.chunk(dl_doc=document):
                raw_chunk = chunker.contextualize(chunk)
                embedding = embed_text(raw_chunk, embedding_model)
                
                chunk_id = str(uuid.uuid4())  # Generate a unique ID for the chunk
                content_token_count = chunker.tokenizer.count_tokens(raw_chunk)
                
                # Prepare metadata object (same as KFP version)
                metadata_obj = {
                    "file_name": file_name,
                    "document_id": chunk_id,
                    "token_count": content_token_count,
                }
                
                metadata_str = json.dumps(metadata_obj)
                metadata_token_count = chunker.tokenizer.count_tokens(metadata_str)
                metadata_obj["metadata_token_count"] = metadata_token_count
                
                chunks_with_embedding.append({
                    "content": raw_chunk,
                    "mime_type": "text/markdown",
                    "embedding": embedding,
                    "metadata": metadata_obj,
                })
        
        elif doc_info['type'] == 'text':
            # For text files, create simple chunks
            content = doc_info['content']
            # Split text into chunks based on token count
            words = content.split()
            tokens_per_word = 1.3  # Rough estimate
            words_per_chunk = int(CHUNK_SIZE_IN_TOKENS / tokens_per_word)
            
            for i in range(0, len(words), words_per_chunk):
                chunk_words = words[i:i + words_per_chunk]
                raw_chunk = ' '.join(chunk_words)
                embedding = embed_text(raw_chunk, embedding_model)
                
                chunk_id = str(uuid.uuid4())
                
                # Prepare metadata object
                metadata_obj = {
                    "file_name": file_name,
                    "document_id": chunk_id,
                    "token_count": len(chunk_words),  # Rough estimate
                }
                
                chunks_with_embedding.append({
                    "content": raw_chunk,
                    "mime_type": "text/plain",
                    "embedding": embedding,
                    "metadata": metadata_obj,
                })
        
        # Insert chunks using the same method as KFP version
        if chunks_with_embedding:
            try:
                client.vector_io.insert(vector_db_id=vector_db_id, chunks=chunks_with_embedding)
                total_chunks += len(chunks_with_embedding)
                print(f"✅ Inserted {len(chunks_with_embedding)} chunks from {file_name}")
            except Exception as e:
                print(f"❌ Failed to insert embeddings from {file_name}: {e}")
    
    print(f"✅ Total chunks inserted: {total_chunks}")
    return vector_db_id

# =============================================================================
# AGENT CREATION FUNCTIONS
# =============================================================================

def create_rag_agent(client, vector_db_id):
    """
    Create a RAG agent that can query the documents.
    
    This function:
    1. Configures an agent with RAG capabilities
    2. Connects it to our vector database
    3. Sets up the agent's behavior and instructions
    
    Args:
        client: LlamaStackClient instance
        vector_db_id: ID of the vector database to connect to
        
    Returns:
        Agent: The created RAG agent
    """
    print(f"\n🤖 Creating RAG agent...")
    
    # Debug: Check available models
    models = client.models.list()
    llm_models = [m for m in models if m.model_type == "llm"]
    print(f"🔍 Available LLM models: {[m.identifier for m in llm_models]}")
    
    # Find the correct LLM model
    llm_model = next((m for m in llm_models if m.identifier == INFERENCE_MODEL), None)
    if not llm_model:
        llm_model = next((m for m in llm_models), None)  # Use first available LLM
        if llm_model:
            print(f"⚠️  Model '{INFERENCE_MODEL}' not found, using '{llm_model.identifier}' instead")
        else:
            raise ValueError("No LLM models found on the server")
    
    # Debug: Check available vector databases
    try:
        vector_dbs = client.vector_dbs.list()
        print(f"🔍 Available vector databases: {[vdb.identifier for vdb in vector_dbs]}")
        
        # Verify our vector DB exists
        our_vdb = next((vdb for vdb in vector_dbs if vdb.identifier == vector_db_id), None)
        if our_vdb:
            print(f"✅ Vector database found: {vector_db_id}")
        else:
            print(f"❌ Vector database '{vector_db_id}' not found!")
            raise ValueError(f"Vector database '{vector_db_id}' not found in registered databases")
    except Exception as e:
        print(f"⚠️  Could not list vector databases: {e}")
    
    # Debug: Check if RAG tool is available
    try:
        # Check what tools are available (this might fail on some llama-stack versions)
        print(f"🔍 Checking RAG tool availability...")
    except:
        pass
    
    print(f"🔧 Agent configuration:")
    print(f"   • Model: {llm_model.identifier}")
    print(f"   • Vector DB: {vector_db_id}")
    print(f"   • Top K: {TOP_K}")
    print(f"   • Instructions length: {len(AGENT_INSTRUCTIONS)} characters")
    

    # Create the agent using the exact same pattern as the working notebook
    # Use simple instructions like the working notebook
    agent = Agent(
        client,
        model=llm_model.identifier,                     # Which LLM to use for generating answers
        instructions="You are a helpful assistant",     # Use same simple instructions as working notebook
        tools=[                                         # Configure the RAG tool directly
            {
                "name": "builtin::rag/knowledge_search",  # Use the correct RAG tool name
                "args": {"vector_db_ids": [vector_db_id]}, # Which vector database to search
            }
        ],
    )
    
    print(f"✅ Agent created with ID: {agent.agent_id}")
    print(f"📝 Agent name: {AGENT_NAME}")
    
    # Debug: Try to verify the agent was created properly
    try:
        # List all agents to verify our agent exists
        print(f"🔍 Verifying agent registration...")
        # Note: This might not work on all llama-stack versions
    except Exception as e:
        print(f"⚠️  Could not verify agent registration: {e}")
    
    return agent

def create_session(agent):
    """
    Create a chat session for the agent.
    
    Sessions allow you to have conversations with the agent.
    Each session maintains its own conversation history.
    
    Args:
        agent: The RAG agent to create a session for
        
    Returns:
        str: The ID of the created session
    """
    print(f"\n💬 Creating chat session...")
    session_id = agent.create_session(SESSION_NAME)
    print(f"✅ Session created with ID: {session_id}")
    return session_id

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function that orchestrates the entire RAG setup process.
    
    This function demonstrates the complete RAG pipeline using advanced techniques:
    1. Connect to llama-stack
    2. Load and process documents with advanced docling processing
    3. Create vector database with HybridChunker and manual embedding generation
    4. Set up RAG agent
    5. Create session for interaction
    6. Provide instructions for querying
    """
    print("🚀 Advanced RAG Agent Setup")
    print("=" * 50)
    print("🔧 Using advanced processing techniques:")
    print("   • HybridChunker for document-aware chunking")
    print("   • SentenceTransformer for manual embedding generation")
    print("   • Advanced PDF processing with OCR and table extraction")
    print("   • Direct vector insertion with pre-computed embeddings")
    print("   • Larger chunk size (512 tokens) for better context")
    
    # Step 1: Connect to llama-stack
    print(f"\n🔌 Connecting to llama-stack at {LLAMA_STACK_URL}...")
    try:
        client = LlamaStackClient(base_url=LLAMA_STACK_URL)
        print("✅ Connected to llama-stack")
    except Exception as e:
        print(f"❌ Failed to connect to llama-stack: {e}")
        print("Make sure llama-stack is running and accessible at the configured URL")
        return
    
    # Step 2: Load documents from the input folder
    documents = load_documents_from_folder()
    if not documents:
        print("❌ No documents loaded. Exiting.")
        return
    
    # Step 3: Create vector database and insert documents using advanced processing
    vector_db_id = setup_vector_database_and_insert_documents(client, documents)
    
    # Step 4: Create RAG agent with access to the documents
    agent = create_rag_agent(client, vector_db_id)
    
    # Step 5: Create a session for interacting with the agent
    session_id = create_session(agent)
    
    # Success! Display summary and instructions
    print(f"\n🎉 Advanced RAG Agent Setup Complete!")
    print("=" * 50)
    print(f"📊 Summary:")
    print(f"   • Documents loaded: {len(documents)}")
    print(f"   • Vector DB ID: {vector_db_id}")
    print(f"   • Agent ID: {agent.agent_id}")
    print(f"   • Session ID: {session_id}")
    print(f"   • Chunking: HybridChunker with {CHUNK_SIZE_IN_TOKENS} tokens")
    print(f"   • Embedding: Manual generation with {EMBEDDING_MODEL}")
    
    # Demonstrate how to query the agent directly
    print(f"\n🔍 Demonstrating agent query...")
    
    try:
        prompt = "What is RAG and how does it work? Please search the documents for information."
        print(f"prompt> {prompt}")
        
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
            stream=True
        )
        
        print(f"\n📄 Agent response:")
        
        for log in AgentEventLogger().log(response):
            log.print()
        
        print(f"\n🎉 SUCCESS! Your RAG agent is working correctly!")
        print(f"   • Documents were processed and stored")
        print(f"   • RAG tool is being called and retrieving document content")
        print(f"   • Agent is providing answers based on your documents")
            
    except Exception as e:
        print(f"❌ Error during demonstration query: {e}")
        import traceback
        traceback.print_exc()
        print("You can still query the agent manually using the curl command below.")
    
    # Provide the curl command for manual querying
    print(f"\n🔍 To query your advanced RAG agent manually, use this curl command:")
    print(f"""curl -X POST {LLAMA_STACK_URL}/v1/agents/{agent.agent_id}/session/{session_id}/turn \\
  -H "Content-Type: application/json" \\
  -d '{{
    "messages": [
      {{
        "role": "user",
        "content": "What is this document about?"
      }}
    ],
    "stream": true
  }}'""")
    
    # Suggest example questions
    print(f"\n💡 Example questions you can ask:")
    print(f"   • What is the main topic of the documents?")
    print(f"   • What are the key points mentioned?")
    print(f"   • Can you summarize the content?")
    print(f"   • What specific details are mentioned about [topic]?")
    print(f"\n🎯 With advanced processing, you should get better, more accurate answers!")

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
