# Benchmarking Llama Stack with the BEIR Framework

## Purpose
The purpose of this script is to provide a variety of different benchmarks users can run with Llama Stack using standardized information retrieval benchmarks from the [BEIR](https://github.com/beir-cellar/beir) framework.

## Available Benchmarks
Currently there is only one benchmark available:
1. [Benchmarking embedding models with BEIR Datasets and Llama Stack](benchmarking_embedding_models.md)


## Prerequisites
* [Python](https://www.python.org/downloads/) > v3.12
* [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) installed
* [ollama](https://ollama.com/) set up on your system and running the `meta-llama/Llama-3.2-3B-Instruct` model

> [!NOTE]
> Ollama can be replaced with an [inference provider](https://llama-stack.readthedocs.io/en/latest/providers/inference/index.html) of your choice

## Installation

Initialize a virtual environment:
``` bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
```

Install the required dependencies:

```bash
uv pip install -r requirements.txt
```

Prepare your environment by running:
``` bash
# The run.yaml file is based on starter template https://github.com/meta-llama/llama-stack/tree/main/llama_stack/templates/starter
# We run a build here to install all of the dependencies for the starter template
llama stack build --template starter --image-type venv
```

## Quick Start

1. **Run a basic benchmark**:
```bash
# Runs the embedding models benchmark by default
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py --dataset-names scifact --embedding-models granite-embedding-125m
```

2. **View results**: Results will be saved in the `results/` directory with detailed evaluation metrics.

## File Structure

```
beir-benchmarks/
├── README.md                          # This file
├── beir_benchmarks.py                 # Main benchmarking script for multiple benchmarks
├── benchmarking_embedding_models.md   # Detailed documentation and guide
├── requirements.txt                   # Python dependencies
└── run.yaml                           # Llama Stack configuration
```

## Usage Examples

### Basic Usage
```bash
# Run benchmark with default settings
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py

# Specify custom dataset and model
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py --dataset-names scifact --embedding-models granite-embedding-125m

# Run with custom batch size
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py --batch-size 100
```

### Advanced Configuration
For advanced configuration options and detailed setup instructions, see [benchmarking_embedding_models.md](benchmarking_embedding_models.md).

## Results

Benchmark results are automatically saved in the `results/` directory in TREC evaluation format. Each result file contains:
- Query-document relevance scores
- Ranking information for retrieval evaluation
- Timestamp and model information in the filename

## Support

For detailed technical documentation, refer to:
- [benchmarking_embedding_models.md](benchmarking_embedding_models.md) - Comprehensive guide for embedding models benchmark
- [BEIR Documentation](https://github.com/beir-cellar/beir) - Official BEIR framework docs
- [Llama Stack Documentation](https://llama-stack.readthedocs.io/) - Llama Stack API reference
