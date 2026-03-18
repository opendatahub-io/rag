# Simple RAG Agent Demo

A didactic example for **facilitating the creation of RAG agents in llama-stack**. This demo provides a streamlined approach to quickly deploy agents with RAG capabilities using PDF and TXT documents as inputs, making it ideal for development lifecycle workflows.

## 🚀 Recent Improvements - Advanced RAG Processing

This script has been enhanced with advanced processing techniques based on the high-performance KFP pipeline implementation:

### ✨ **Enhanced Document Processing**
- **HybridChunker**: Uses document-aware chunking instead of simple token-based splitting
- **Advanced PDF Processing**: Enables OCR, table structure extraction, and page image generation
- **Better Text Extraction**: Preserves document structure and metadata

### 🧠 **Improved Embedding Generation**
- **Manual Embedding Control**: Uses SentenceTransformer for direct embedding generation
- **Larger Chunk Size**: Increased from 256 to 512 tokens for better context
- **Rich Metadata**: Includes token counts, document IDs, and processing information

### 🗄️ **Advanced Vector Database Operations**
- **Direct Vector Insertion**: Uses `client.vector_io.insert()` for pre-computed embeddings
- **Better Chunk Management**: Contextualizes chunks with document structure
- **Enhanced Metadata**: Detailed tracking of document processing statistics

### 🎯 **Performance Benefits**
- **Better Retrieval Quality**: Document-aware chunking preserves semantic meaning
- **Improved Answer Accuracy**: Larger context windows provide more comprehensive answers
- **Enhanced PDF Support**: OCR and table extraction handle complex documents better

These improvements should provide significantly better RAG results compared to the basic version.

## Purpose

This simple RAG script is designed to **facilitate the development lifecycle** by providing a quick and easy way to:
- **Deploy agents rapidly** with RAG capabilities
- **Process documents** (PDF and TXT) for knowledge base creation
- **Create vector databases** automatically from your documents
- **Set up AI agents** that can answer questions based on your specific documents
- **Streamline the development process** for RAG-enabled applications

## What is RAG?

Retrieval Augmented Generation (RAG) is a technique that combines:
1. **Document Retrieval**: Finding relevant information from a knowledge base
2. **Text Generation**: Using an AI model to generate answers based on the retrieved information

This approach helps AI models provide more accurate and up-to-date answers by grounding their responses in specific documents.

## Development Lifecycle Benefits

This script is particularly useful for:

### 🚀 **Rapid Prototyping**
- Quickly test RAG concepts with your documents
- Iterate on agent configurations without complex setup
- Validate document processing pipelines

### 🔄 **Development Workflow**
- Easy integration into CI/CD pipelines
- Consistent agent creation across environments
- Simplified testing of RAG functionality

### 📚 **Document Processing**
- Automated handling of PDF and TXT files
- Built-in text extraction and chunking
- Vector database setup without manual configuration

### 🤖 **Agent Deployment**
- One-command agent creation
- Configurable agent parameters
- Ready-to-use chat sessions

## How This Demo Works

The script demonstrates these simple steps:

1. **📁 Load Documents**: Read text and PDF files from the `input_files` folder
2. **🔄 Convert to Text**: Extract text content from different file formats
3. **🗄️ Store in Vector DB**: Save documents in a searchable vector database
4. **🤖 Create Agent**: Set up an AI agent that can query the documents
5. **💬 Ask Questions**: Query the agent to get answers based on your documents

## Prerequisites

- Python 3.8+
- A running llama-stack instance (see setup below)
- Some text or PDF files to process

## Setup

### 1. Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Start llama-stack

Make sure you have llama-stack running and accessible. You can use port-forwarding to access it locally:

```bash
# If running on OpenShift
oc port-forward svc/lsd-llama-milvus 8081:8081

# Or if running locally
# Follow llama-stack installation instructions
```

### 3. Add Your Documents

Place your text (`.txt`) and PDF (`.pdf`) files in the `input_files` folder:

```
input_files/
├── document1.txt
├── document2.pdf
└── ...
```

## Usage

### Run the RAG Setup

```bash
python setup_rag_agent.py
```

The script will:
- Load all documents from `input_files/`
- Create a vector database
- Set up a RAG agent
- Provide you with the IDs and a curl command to query the agent

### Query Your RAG Agent

After running the script, you'll get a curl command like this:

```bash
curl -X POST http://localhost:8081/v1/agents/{agent_id}/session/{session_id}/turn \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What is this document about?"
      }
    ],
    "stream": true
  }'
```

### Example Questions

Try asking questions like:
- "What is the main topic of the documents?"
- "What are the key points mentioned?"
- "Can you summarize the content?"
- "What specific details are mentioned about [topic]?"

## Configuration

You can modify these settings at the top of `setup_rag_agent.py`:

### Basic Settings
```python
LLAMA_STACK_URL = "http://localhost:8081"  # Your llama-stack URL
INFERENCE_MODEL = "vllm"                   # Model for generating answers
EMBEDDING_MODEL = "granite-embedding-125m"  # Model for embeddings
AGENT_NAME = "Simple RAG Agent"            # Custom name for your agent
```

### Document Processing
```python
INPUT_FOLDER = "input_files"               # Folder containing your documents
SUPPORTED_EXTENSIONS = [".txt", ".pdf"]    # File types to process
CHUNK_SIZE_IN_TOKENS = 256                 # Size of text chunks for vector database
```

### Vector Database
```python
VECTOR_DB_PROVIDER = "milvus"              # Vector database provider
VECTOR_DB_PREFIX = "simple-rag-db"         # Prefix for vector database ID
```

### RAG Agent Settings
```python
TOP_K = 3                                  # Number of most relevant chunks to retrieve
SIMILARITY_THRESHOLD = 0.0                 # Minimum similarity score for retrieval
MAX_INFER_ITERS = 10                       # Maximum inference iterations
ENABLE_SESSION_PERSISTENCE = False         # Whether to persist sessions
```

### PDF Processing
```python
PDF_DO_OCR = False                         # Whether to perform OCR on PDFs
PDF_DO_TABLE_STRUCTURE = True              # Whether to extract table structures
PDF_DO_CELL_MATCHING = True                # Whether to perform cell matching in tables
```

### Session & Logging
```python
SESSION_NAME = "simple-rag-session"        # Name for the chat session
LOG_LEVEL = "INFO"                         # Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Agent Instructions
```python
AGENT_INSTRUCTIONS = """You are a helpful assistant..."""  # Custom instructions for the agent
```

## Supported File Types

- **Text files** (`.txt`): Plain text documents
- **PDF files** (`.pdf`): PDF documents with text extraction and table structure

## Troubleshooting

### Connection Issues
- Make sure llama-stack is running and accessible
- Check the `LLAMA_STACK_URL` configuration
- Verify port-forwarding is working

### Document Processing Issues
- Ensure files are in supported formats (`.txt`, `.pdf`)
- Check file permissions and encoding
- For PDFs, make sure they contain extractable text

### Model Issues
- Verify the specified models are available in your llama-stack
- Check model names match exactly

## Understanding the Code

The script is structured in simple, clear functions:

- `load_text_file()`: Reads plain text files
- `load_pdf_file()`: Extracts text from PDFs using docling
- `load_documents_from_folder()`: Processes all files in the input folder
- `setup_vector_database()`: Creates and populates the vector database
- `create_rag_agent()`: Sets up the AI agent with RAG capabilities
- `create_session()`: Creates a chat session for the agent

Each function has a single responsibility and clear error handling, making it easy to understand and modify.

## Next Steps

Once you understand this basic RAG setup, you can explore:

### 🔧 **Development Enhancements**
- **Custom agent configurations** for specific use cases
- **Advanced document processing** pipelines
- **Integration with CI/CD** for automated agent deployment
- **Environment-specific configurations** (dev, staging, prod)

### 🚀 **Production Deployment**
- **Web interface** for agent management
- **API endpoints** for programmatic agent creation
- **Monitoring and logging** for agent performance
- **Scalable vector database** configurations

### 📊 **Advanced Features**
- **Custom retrieval strategies** for better document matching
- **Multi-modal document support** (images, audio, etc.)
- **Real-time document updates** and agent retraining
- **Performance optimization** for large document sets

### 🔗 **Integration Possibilities**
- **Chatbot interfaces** for end users
- **Knowledge management systems**
- **Documentation assistants**
- **Customer support automation**
