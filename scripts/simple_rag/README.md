# RAG Agent Script

This script creates a RAG (Retrieval Augmented Generation) agent using Llama Stack that can answer questions based on uploaded documents.

## Prerequisites

1. **OpenShift Llama Stack deployed** - Make sure you have the Llama Stack service running on OpenShift
2. **Python environment** - Python 3.8+ with virtual environment
3. **Document files** - Text files to upload (place in `input_files` folder)

## Setup

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your documents:**
   - Add your `.txt` files to the `input_files` folder
   - The script will automatically process all `.txt` files in this folder
   - An example file (`example.txt`) is already included

## Usage

### Step 1: Run the Script
```bash
python rag_agent.py
```

The script will:
1. Scan the `input_files` folder for all `.txt` files
2. Connect to the remote Llama Stack service
3. Create a vector database
4. Upload and process all found documents
5. Create a RAG agent with the combined document knowledge
6. Create a session for the agent
7. Print the curl command for querying

### Step 2: Query the Agent
After running the script, you'll see output like:
```
📁 Found 2 .txt file(s): ['example.txt', 'document2.txt']
📄 Reading example.txt...
📄 Reading document2.txt...
📥 Inserting documents into vector DB...
📄 Inserted 2 document(s) with approx 150 words total.

✅ RAG Agent Setup Complete!

📋 Query Information:
Vector DB ID: multi-doc-vector-db-xxxxx
Agent ID: agent-id-xxxxx
Session ID: session-id-xxxxx

🔍 To query the RAG agent, use this curl command:
curl -X POST http://lsd-llama-milvus-rag-stack.apps.rosa.e8a7l7t5a1a1z2m.cusw.p3.openshiftapps.com/v1/agents/agent-id/session/session-id/turn \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "What information do you have about AI research?"
      }
    ],
    "stream": true
  }'
```

### Step 3: Use the Curl Command
Copy and paste the curl command from the script output. You can modify the question in the `content` field to ask different questions about any content from your uploaded documents.

## Example Questions
Once your agent is set up, you can ask questions about any content from your documents:
- "What information do you have about [topic]?"
- "Summarize the key points from the documents"
- "Tell me about [specific person/concept] mentioned in the files"
- "What are the main themes across all documents?"

## Adding More Documents
To add more documents to your RAG agent:
1. Place additional `.txt` files in the `input_files` folder
2. Run the script again (this creates a new agent with all files)
3. The new agent will have knowledge from all documents combined

## Configuration

You can modify the script to:
- **Change the input folder:** Edit the `input_folder` path in the script
- **Update the Llama Stack URL:** Change `LLAMA_STACK_URL` if your service is at a different location
- **Adjust chunk size:** Modify `chunk_size_in_tokens` parameter (default: 256)
- **Change retrieval settings:** Adjust `top_k` (default: 3) or `similarity_threshold` (default: 0.0)

## Troubleshooting

### Common Issues

1. **No Files Found:**
   ```
   ❌ No .txt files found in the input_files folder!
   ```
   - Make sure you have `.txt` files in the `input_files` folder
   - Check that the files have the `.txt` extension

2. **Connection Error:**
   ```
   ❌ Error occurred: HTTPConnectionPool(host='...', port=80): Max retries exceeded
   ```
   - Check if the Llama Stack service is running
   - Verify the URL is correct

3. **Module Not Found:**
   ```
   ModuleNotFoundError: No module named 'llama_stack_client'
   ```
   - Make sure you installed dependencies: `pip install -r requirements.txt`

4. **Empty Response:**
   - Make sure your documents contain relevant information
   - Try asking more specific questions
   - Check if the documents were uploaded successfully (look for "📄 Inserted X document(s)")

### Debug Mode
To see detailed API calls, the script already includes debug logging. You'll see HTTP requests in the output showing exactly what's happening.

## Files Structure
```
scripts/simple_rag/
├── rag_agent.py        # Main script
├── requirements.txt    # Python dependencies
├── input_files/        # Folder for your documents
│   └── example.txt     # Example document
└── README.md          # This file
```

## Notes
- Each run of the script creates a new agent and vector database
- The agent combines knowledge from ALL `.txt` files in the input_files folder
- The agent stays alive on the OpenShift cluster until manually deleted
- You can query the same agent multiple times using the curl command
- The script uses streaming responses for real-time output
- Documents are processed with metadata including filename for better tracking
