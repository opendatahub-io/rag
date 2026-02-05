# Qdrant Vector IO Demo

Demonstrates Qdrant as a vector store provider through Llama Stack, using the OpenAI-compatible API for all operations.

## What's Covered

| Notebook | Topics |
|----------|--------|
| [01 Search Modes](notebooks/01_search_modes_demo.ipynb) | Vector, keyword, and hybrid search; score thresholds; vector store CRUD |
| [02 Multilingual](notebooks/02_multilingual_demo.ipynb) | Cross-lingual search (EN/ES/IT); keyword vs vector comparison |
| [03 Multimodal](notebooks/03_multimodal_demo.ipynb) | Text-to-image search via captions; keyword tag search |
| [04 Advanced Features](notebooks/04_advanced_features_demo.ipynb) | Metadata filtering (eq/ne/gt/lt/and/or); chunking strategies |

### Search Modes

| Mode | How It Works | Qdrant Implementation |
|------|--------------|----------------------|
| **Vector** | Cosine similarity on embeddings | `query_points` |
| **Keyword** | Splits query into words, matches any via `MatchText` | `scroll` with filter |
| **Hybrid** | Vector similarity filtered by keyword matches | `query_points` with `query_filter` |

## Prerequisites

| Component | Purpose | Setup |
|-----------|---------|-------|
| **Ollama** | Embedding model + LLM | `ollama serve` then `ollama pull nomic-embed-text` |
| **Qdrant** | Vector database | `podman run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant` |
| **Llama Stack** | API server with Qdrant provider | See below |

### Start Llama Stack with Qdrant

```bash
OLLAMA_URL=http://localhost:11434/v1 \
QDRANT_URL=http://localhost:6333 \
llama stack run starter --port 8321
```

Verify the Qdrant provider is active:

```bash
curl -s http://localhost:8321/v1/providers | python3 -c "
import json, sys
for p in json.load(sys.stdin).get('data', []):
    if 'qdrant' in p.get('provider_id', ''):
        print(f\"  {p['provider_id']}: {p['provider_type']}\")
"
```

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Running the Notebooks

```bash
cd demos/qdrant-vector-io
jupyter notebook notebooks/
```

Run them in order (01 through 04). Each notebook is self-contained: it creates a vector store, inserts data, demonstrates features, and cleans up.

## Data

| File | Size | Description |
|------|------|-------------|
| `data/startups_demo.json` | 500 records | Startup company descriptions (JSON Lines) |
| `data/multilingual/` | 6 files | Articles in English, Spanish, Italian |
| `data/images/` | 5 images + captions | Sample images with multilingual captions |

The startups dataset is a 500-record subset. The notebooks use 25-50 records per demo.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────┐
│  Jupyter     │────▶│  Llama Stack  │────▶│  Qdrant │
│  Notebooks   │     │  (port 8321)  │     │ (6333)  │
│              │     │               │     │         │
│  OpenAI      │     │  ┌──────────┐ │     │ Vector  │
│  Python SDK  │     │  │ Ollama   │ │     │ Store   │
│              │     │  │ nomic-   │ │     │         │
└─────────────┘     │  │ embed    │ │     └─────────┘
                    │  └──────────┘ │
                    └──────────────┘
```

All operations use the **OpenAI-compatible API** via the standard `openai` Python client pointed at the Llama Stack server.

## Key API Patterns

### Create a vector store

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8321/v1/", api_key="none")

store = client.vector_stores.create(
    name="my_store",
    extra_body={
        "provider_id": "qdrant",
        "embedding_model": "ollama/nomic-embed-text:latest"
    }
)
```

### Search with different modes

```python
results = client.vector_stores.search(
    vector_store_id=store.id,
    query="search query",
    max_num_results=5,
    ranking_options={"score_threshold": 0.5},
    extra_body={"search_mode": "hybrid"}  # "vector", "keyword", or "hybrid"
)
```

### Search with metadata filters

```python
results = client.vector_stores.search(
    vector_store_id=store.id,
    query="search query",
    filters={"type": "eq", "key": "category", "value": "software"},
    extra_body={"search_mode": "vector"}
)
```
