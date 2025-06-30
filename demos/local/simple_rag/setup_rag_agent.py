#!/usr/bin/env python3
"""
Simple RAG Agent Script - A didactic example of Retrieval Augmented Generation

This script demonstrates the basic steps of RAG:
1. Load documents from files
2. Convert them to text
3. Store them in a vector database
4. Create an agent that can query the documents
5. Ask questions and get answers

Usage:
    python setup_rag_agent.py
"""

import uuid
from pathlib import Path
import logging
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================
# All the parameters you can customize for your RAG setup

# Basic connection and model settings
LLAMA_STACK_URL = "http://localhost:8081"  # URL where your llama-stack is running
INFERENCE_MODEL = "vllm"                   # Model used to generate answers (LLM)
EMBEDDING_MODEL = "granite-embedding-125m"  # Model used to create embeddings (converts text to vectors)
EMBEDDING_DIM = 768                        # Dimension of the embedding vectors
AGENT_NAME = "Simple RAG Agent"            # Human-readable name for your agent

# Document processing settings
INPUT_FOLDER = "input_files"               # Folder where your documents are stored
SUPPORTED_EXTENSIONS = [".txt", ".pdf"]    # File types this script can process
CHUNK_SIZE_IN_TOKENS = 256                 # How to split documents into chunks for better retrieval

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

# PDF processing options (how to handle PDF files)
PDF_DO_OCR = False                         # Use OCR to extract text from images in PDFs (slower but more comprehensive)
PDF_DO_TABLE_STRUCTURE = True              # Extract table structures from PDFs
PDF_DO_CELL_MATCHING = True                # Match table cells for better table understanding

# Logging configuration
LOG_LEVEL = "INFO"                         # How detailed the logging should be

# Enable logging to see what's happening during execution
# Options: DEBUG, INFO, WARNING, ERROR
logging.basicConfig(level=getattr(logging, LOG_LEVEL))

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
    Load a PDF file and extract its text content using docling.
    
    This function uses advanced PDF processing to:
    - Extract text from PDF documents
    - Preserve table structures
    - Handle complex layouts
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: The extracted text content, or None if there's an error
    """
    print(f"📄 Loading PDF file: {file_path.name}")
    try:
        # Configure PDF processing options using our variables
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = PDF_DO_OCR  # Whether to use OCR for image-based text
        pdf_options.do_table_structure = PDF_DO_TABLE_STRUCTURE  # Extract table structures
        if PDF_DO_TABLE_STRUCTURE:
            pdf_options.table_structure_options.do_cell_matching = PDF_DO_CELL_MATCHING  # Match table cells
        
        # Create a document converter with our PDF settings
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options,
                    backend=PyPdfiumDocumentBackend,  # Use PyPdfium for robust PDF processing
                ),
            },
        )
        
        # Convert the PDF to text
        result = converter.convert(file_path)
        if result and result.document:
            content = result.document.export_to_text()
            return content.strip()
        else:
            print(f"❌ Could not extract text from {file_path.name}")
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
    3. Creates Document objects for llama-stack
    4. Returns a list of processed documents
    
    Args:
        folder_path: Path to the folder containing documents
        
    Returns:
        list: List of Document objects ready for vector database insertion
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
        elif file_path.suffix.lower() == ".pdf":
            content = load_pdf_file(file_path)
        
        if content:
            # Create a Document object that llama-stack can understand
            document = Document(
                document_id=f"doc-{uuid.uuid4().hex}",  # Unique ID for this document
                content=content,                        # The actual text content
                mime_type="text/plain",                 # Type of content
                metadata={"source": str(file_path)},    # Track where this came from
            )
            documents.append(document)
            print(f"✅ Loaded: {file_path.name} ({len(content)} characters)")
    
    return documents

# =============================================================================
# VECTOR DATABASE FUNCTIONS
# =============================================================================

def setup_vector_database(client, documents):
    """
    Create a vector database and insert documents into it.
    
    This function:
    1. Creates a new vector database in llama-stack
    2. Converts documents to embeddings (vectors)
    3. Stores the embeddings for later retrieval
    
    Args:
        client: LlamaStackClient instance
        documents: List of Document objects to insert
        
    Returns:
        str: The ID of the created vector database
    """
    print(f"\n🗄️  Setting up vector database...")
    
    # Create a unique ID for this vector database
    vector_db_id = f"{VECTOR_DB_PREFIX}-{uuid.uuid4().hex}"
    
    # Register the vector database with llama-stack
    # This tells llama-stack to create a new vector database with our settings
    client.vector_dbs.register(
        vector_db_id=vector_db_id,           # Unique identifier
        embedding_model=EMBEDDING_MODEL,     # Which model to use for embeddings
        embedding_dimension=EMBEDDING_DIM,   # Size of the embedding vectors
        provider_id=VECTOR_DB_PROVIDER       # Type of vector database
    )
    print(f"✅ Vector database registered: {vector_db_id}")
    
    # Insert documents into the vector database
    # This converts text to embeddings and stores them for retrieval
    print("📥 Inserting documents into vector database...")
    client.tool_runtime.rag_tool.insert(
        documents=documents,                    # The documents to insert
        vector_db_id=vector_db_id,             # Which database to use
        chunk_size_in_tokens=CHUNK_SIZE_IN_TOKENS  # How to split documents
    )
    
    # Calculate and display statistics
    total_words = sum(len(doc.content.split()) for doc in documents)
    print(f"✅ Inserted {len(documents)} document(s) with ~{total_words} words")
    
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
    
    # Configure the agent with all our settings
    agent_config = {
        "model": INFERENCE_MODEL,                    # Which LLM to use for generating answers
        "name": AGENT_NAME,                          # Human-readable name
        "instructions": AGENT_INSTRUCTIONS,          # How the agent should behave
        "enable_session_persistence": ENABLE_SESSION_PERSISTENCE,  # Remember conversations?
        "max_infer_iters": MAX_INFER_ITERS,          # Maximum reasoning steps
        
        # Configure the RAG tool (this is what makes it a RAG agent)
        "toolgroups": [
            {
                "name": "builtin::rag",              # Use the built-in RAG tool
                "args": {
                    "vector_db_ids": [vector_db_id], # Which vector database to search
                    "top_k": TOP_K,                  # How many chunks to retrieve
                    "similarity_threshold": SIMILARITY_THRESHOLD  # Minimum similarity
                }
            }
        ]
    }
    
    # Create the agent using llama-stack
    agent = Agent(client, agent_config)
    print(f"✅ Agent created with ID: {agent.agent_id}")
    print(f"📝 Agent name: {AGENT_NAME}")
    
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
    
    This function demonstrates the complete RAG pipeline:
    1. Connect to llama-stack
    2. Load and process documents
    3. Create vector database
    4. Set up RAG agent
    5. Create session for interaction
    6. Provide instructions for querying
    """
    print("🚀 Simple RAG Agent Setup")
    print("=" * 50)
    
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
    
    # Step 3: Create vector database and insert documents
    vector_db_id = setup_vector_database(client, documents)
    
    # Step 4: Create RAG agent with access to the documents
    agent = create_rag_agent(client, vector_db_id)
    
    # Step 5: Create a session for interacting with the agent
    session_id = create_session(agent)
    
    # Success! Display summary and instructions
    print(f"\n🎉 RAG Agent Setup Complete!")
    print("=" * 50)
    print(f"📊 Summary:")
    print(f"   • Documents loaded: {len(documents)}")
    print(f"   • Vector DB ID: {vector_db_id}")
    print(f"   • Agent ID: {agent.agent_id}")
    print(f"   • Session ID: {session_id}")
    
    # Provide the curl command for querying
    print(f"\n🔍 To query your RAG agent, use this curl command:")
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

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
