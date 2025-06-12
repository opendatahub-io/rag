import uuid
from pathlib import Path
import glob
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
import logging

# Enable debug logging to see the exact API calls
logging.basicConfig(level=logging.DEBUG)

# OpenShift Llama Stack configuration
LLAMA_STACK_URL = "http://lsd-llama-milvus-rag-stack.apps.rosa.e8a7l7t5a1a1z2m.cusw.p3.openshiftapps.com"
INFERENCE_MODEL = "vllm"
EMBEDDING_MODEL = "ibm-granite/granite-embedding-125m-english"
EMBEDDING_DIM = 768

# Find all .txt files in the input_files folder
input_folder = Path("input_files")
txt_files = list(input_folder.glob("*.txt"))

if not txt_files:
    print("❌ No .txt files found in the input_files folder!")
    print("Please add some .txt files to the input_files folder and try again.")
    exit(1)

print(f"📁 Found {len(txt_files)} .txt file(s): {[f.name for f in txt_files]}")

def create_http_client():
    return LlamaStackClient(
        base_url=LLAMA_STACK_URL
    )

client = create_http_client()

# Read all .txt files from the input_files folder
documents = []
for i, file_path in enumerate(txt_files):
    print(f"📄 Reading {file_path.name}...")
    content = file_path.read_text(encoding="utf-8")
    documents.append(
        Document(
            document_id=f"doc-{i}-{file_path.stem}",
            content=content,
            mime_type="text/plain",
            metadata={"filename": file_path.name}
        )
    )

# Register vector DB in llama-stack
vector_db_id = f"multi-doc-vector-db-{uuid.uuid4().hex}"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model=EMBEDDING_MODEL,
    embedding_dimension=EMBEDDING_DIM,
    provider_id="milvus"
)

print("📥 Inserting documents into vector DB...")
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=256
)
total_words = sum(len(doc.content.split()) for doc in documents)
print(f"📄 Inserted {len(documents)} document(s) with approx {total_words} words total.")

# Create RAG agent
try:
    agent_config = {
        "model": INFERENCE_MODEL,
        "instructions": "You are a helpful assistant that uses only the provided documents. When asked a question, first search the documents and then answer based on what you find.",
        "enable_session_persistence": False,
        "toolgroups": [
            {
                "name": "builtin::rag",
                "args": {
                    "vector_db_ids": [vector_db_id],
                    "top_k": 3,
                    "similarity_threshold": 0.0
                }
            }
        ]
    }
    
    print("\n🔍 Creating agent...")
    rag_agent = Agent(client, agent_config)
    agent_id = rag_agent.agent_id
    print(f"✅ Agent created with ID: {agent_id}")
    
    print("\n🔍 Creating session...")
    session_id = rag_agent.create_session("matias-rag-session")
    print(f"✅ Session created with ID: {session_id}")
    
    print("\n✅ RAG Agent Setup Complete!")
    print("\n📋 Query Information:")
    print(f"Vector DB ID: {vector_db_id}")
    print(f"Agent ID: {agent_id}")
    print(f"Session ID: {session_id}")
    print("\n🔍 To query the RAG agent, use this curl command:")
    print(f"""curl -X POST {LLAMA_STACK_URL}/v1/agents/{agent_id}/session/{session_id}/turn \\
  -H "Content-Type: application/json" \\
  -d '{{
    "messages": [
      {{
        "role": "user",
        "content": "<Your question here>?"
      }}
    ],
    "stream": true
  }}'""")

except Exception as e:
    print(f"\n❌ Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
