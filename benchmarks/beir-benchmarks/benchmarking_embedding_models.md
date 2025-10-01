# Benchmarking embedding models with BEIR Datasets and Llama Stack

## Purpose
The purpose of this script is to compare retrieval accuracy between embedding models using standardized information retrieval benchmarks from the [BEIR](https://github.com/beir-cellar/beir) framework.

## Setup
For the examples we use Ollama to serve the model which can easily be swapped for an inference provider of your choice.

Initialize a virtual environment:
``` bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
```

Install the script's dependencies:
``` bash
uv pip install -r requirements.txt
```

Prepare your environment by running:
``` bash
# The run.yaml file is based on starter distro https://github.com/llamastack/llama-stack/tree/v0.2.23/llama_stack/distributions/starter
# We run a build here to install all of the dependencies for the starter distro
llama stack build --distro starter --image-type venv
```

## Running Instructions

### Basic Usage
To run the script with default settings:

```bash
# This will run the script with the provided run.yaml using LlamaStackAsLibraryClient
MILVUS_URL=milvus uv run python beir_benchmarks.py
```

To run the script with a custom Llama Stack server:

```bash
LLAMA_STACK_URL="http://localhost:8321" MILVUS_URL=milvus uv run python beir_benchmarks.py
```

## Supported Embedding Models

Default supported embedding models:
- `granite-embedding-30m`: IBM Granite 30M parameter embedding model
- `granite-embedding-125m`: IBM Granite 125M parameter embedding model

It is possible to add more embedding models using the [Llama Stack Python Client](https://github.com/llamastack/llama-stack-client-python)

### Adding additional embedding models
Below is an example of how you can add more embedding models to the models list.
``` bash
# First run the llama stack server via the run file
MILVUS_URL=milvus uv run llama stack run run.yaml
```
``` bash
# Adding the all-MiniLM-L6-v2 model via the llama-stack-client
llama-stack-client models register all-MiniLM-L6-v2 --provider-id sentence-transformers --provider-model-id all-minilm:latest --metadata '{"embedding_dimension": 384}' --model-type embedding
```
> [!NOTE]
> Shut down the Llama Stack server before running the benchmark

### Command-Line Options

#### `--vector-db-provider-id`
**Description:** Specifies which vector db provider to use for benchmarking

- **Type:** String
- **Default:** `"milvus"`

**Example:**
```bash
# Set the provider to remote Milvus
--vector-db-provider-id "remote-milvus"
```

#### `--dataset-names`
**Description:** Specifies which BEIR datasets to use for benchmarking.

- **Type:** List of strings
- **Default:** `["scifact"]`
- **Options:** Any dataset from the [available BEIR Datasets](https://github.com/beir-cellar/beir?tab=readme-ov-file#beers-available-datasets)
- **Note:** When using custom datasets (via `--custom-datasets-urls`), this flag provides names for those datasets

**Example:**
```bash
# Single dataset
--dataset-names scifact

# Multiple datasets
--dataset-names scifact scidocs nq
```

#### `--embedding-models`
**Description:** Specifies which embedding models to benchmark against each other.

- **Type:** List of strings
- **Default:** `["granite-embedding-30m", "granite-embedding-125m"]`
- **Requirement:** Embedding models must be defined in the `run.yaml` file
- **Purpose:** Compare performance across different embedding models

**Example:**
```bash
# Default models
--embedding-models granite-embedding-30m granite-embedding-125m

# Custom model selection
--embedding-models all-MiniLM-L6-v2 granite-embedding-125m
```

#### `--custom-datasets-urls`
**Description:** Provides URLs for custom BEIR-compatible datasets instead of using the pre-made BEIR datasets.

- **Type:** List of strings
- **Default:** `[]` (empty - uses standard BEIR datasets)
- **Requirement:** Must be used together with `--dataset-names` flag
- **Format:** URLs pointing to BEIR-compatible dataset archives

**Example:**
```bash
# Using custom datasets
--dataset-names my-custom-dataset --custom-datasets-urls https://example.com/my-dataset.zip
```

#### `--batch-size`
**Description:** Controls the number of documents processed in each batch when injecting documents into the vector database.

- **Type:** Integer
- **Default:** `150`
- **Purpose:** Manages memory usage and processing efficiency when inserting large document collections
- **Note:** Larger batch sizes may be faster but use more memory; smaller batch sizes use less memory but may be slower

**Example:**
```bash
# Using smaller batch size for memory-constrained environments
--batch-size 50

# Using larger batch size for faster processing
--batch-size 300
```

> [!NOTE]
   Your custom Dataset must adhere to the following file structure and document standards. Below is a snippet of the file structure and example documents.

``` text
dataset-name.zip/
├── qrels/
│   └── test.tsv     # Relevance judgments mapping query IDs to document IDs with relevance scores
├── corpus.jsonl     # Document collection with document IDs, titles, and text content
└── queries.jsonl    # Test queries with query IDs and question text for retrieval evaluation
```

**test.tsv**

| query-id | corpus-id | score |
|----------|-----------|-------|
| 0        | 0  | 1     |
| 1        | 1  | 1     |

**corpus.jsonl**
``` json
{"_id": "0", "title": "Hook Lighthouse is located in Wexford, Ireland.", "metadata": {}}
{"_id": "1", "title": "The Captain of the Pequod is Captain Ahab.", "metadata": {}}
```

**queries.jsonl**
``` json
{"_id": "0", "text": "Hook Lighthouse location", "metadata": {}}
{"_id": "1", "text": "Captain of the Pequod", "metadata": {}}
```

### Usage Examples

**Basic benchmarking with default settings:**
```bash
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py
```

**Basic benchmarking with larger batch size:**
```bash
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py --batch-size 300
```

**Benchmark multiple datasets:**
```bash
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py \
 --dataset-names scifact scidocs
```

**Compare specific embedding models:**
```bash
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py \
  --embedding-models granite-embedding-30m all-MiniLM-L6-v2
```

**Use custom datasets:**
```bash
ENABLE_OLLAMA=ollama ENABLE_MILVUS=milvus OLLAMA_INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python beir_benchmarks.py \
  --dataset-names my-dataset \
  --custom-datasets-urls https://example.com/my-beir-dataset.zip
```

### Sample Output
Below is sample outputs for the following datasets:
* scifact
* fiqa
* arguana

> [!NOTE]
   Benchmarking with these datasets will take a considerable amount of time given that fiqa and arguana are much larger and take longer to ingest.

```
Scoring
All results in <path-to>/rag/benchmarks/embedding-models-with-beir/results

scifact map_cut_10
 granite-embedding-125m                            :     0.6879
 granite-embedding-30m                             :     0.6578
 p_value                                           :     0.0150
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


scifact map_cut_5
 granite-embedding-125m                            :     0.6767
 granite-embedding-30m                             :     0.6481
 p_value                                           :     0.0294
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


scifact ndcg_cut_10
 granite-embedding-125m                            :     0.7350
 granite-embedding-30m                             :     0.7018
 p_value                                           :     0.0026
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


scifact ndcg_cut_5
 granite-embedding-125m                            :     0.7119
 granite-embedding-30m                             :     0.6833
 p_value                                           :     0.0256
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


fiqa map_cut_10
 granite-embedding-125m                            :     0.3581
 granite-embedding-30m                             :     0.2829
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


fiqa map_cut_5
 granite-embedding-125m                            :     0.3395
 granite-embedding-30m                             :     0.2664
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


fiqa ndcg_cut_10
 granite-embedding-125m                            :     0.4411
 granite-embedding-30m                             :     0.3599
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


fiqa ndcg_cut_5
 granite-embedding-125m                            :     0.4176
 granite-embedding-30m                             :     0.3355
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


arguana map_cut_10
 granite-embedding-125m                            :     0.2927
 granite-embedding-30m                             :     0.2821
 p_value                                           :     0.0104
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


arguana map_cut_5
 granite-embedding-125m                            :     0.2707
 granite-embedding-30m                             :     0.2594
 p_value                                           :     0.0216
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


arguana ndcg_cut_10
 granite-embedding-125m                            :     0.4251
 granite-embedding-30m                             :     0.4124
 p_value                                           :     0.0044
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort


arguana ndcg_cut_5
 granite-embedding-125m                            :     0.3718
 granite-embedding-30m                             :     0.3582
 p_value                                           :     0.0292
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m generation is better on data of this sort
```
