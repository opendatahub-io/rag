# Benchmarking Information Retrieval with and without Llama Stack

## Purpose
The purpose of this script is to benchmark retrieval accuracy with and without Llama Stack using the BEIR IR or BEIR like datasets.
If everything is working as intended, it will show no difference with and without Llama Stack.
In contrast, if there is some sort of defect in either Llama Stack or the alternative implementation this benchmark should be able to showcase it.

## Setup Instructions
Ollama is required to run this example with the provided [run.yaml](run.yaml) file.

Setup a virtual environment:
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
llama stack build --template ollama --image-type venv
```

### About the run.yaml file
* The run.yaml file makes use of Milvus inline as its vector database. 
* There are 3 default embedding models `ibm-granite/granite-embedding-125m-english`, `ibm-granite/granite-embedding-30m-english` and `all-MiniLM-L6-v2`.

To add your own embedding models you can update the `models` section of the `run.yaml` file.
``` yaml
# Example adding <example-model> embedding model with sentence-transformers as its provider
models:
- metadata: {}
  model_id: ${env.INFERENCE_MODEL}
  provider_id: ollama
  model_type: llm
- metadata:
    embedding_dimension: 768
  model_id: granite-embedding-125m
  provider_id: sentence-transformers
  provider_model_id: ibm-granite/granite-embedding-125m-english
  model_type: embedding
- metadata:
    embedding_dimension: <int>
  model_id: <example-model>
  provider_id: sentence-transformers
  provider_model_id: sentence-transformers/<example-model>
  model_type: embedding
```


## Running Instructions

### Basic Usage
To run the script with default settings:

```bash
# Update INFERENCE_MODEL to your preferred model served by Ollama
INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python benchmark_beir_ls_vs_no_ls.py
```

### Command-Line Options

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
INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python benchmark_beir_ls_vs_no_ls.py
```

**Basic benchmarking with larger batch size:**
```bash
INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python benchmark_beir_ls_vs_no_ls.py --batch-size 300
```

**Benchmark multiple datasets:**
```bash
INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python benchmark_beir_ls_vs_no_ls.py \
 --dataset-names scifact scidocs
```

**Use custom datasets:**
```bash
INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" uv run python benchmark_beir_ls_vs_no_ls.py \
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
scifact map_cut_10
 LlamaStackRAGRetriever                            :     0.6879
 MilvusRetriever                                   :     0.6879
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 300 questions which typically produces a margin of error of around +/-5.8%.
  So the two are probably roughly within that margin of error or so.


scifact ndcg_cut_10
 LlamaStackRAGRetriever                            :     0.7350
 MilvusRetriever                                   :     0.7350
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 300 questions which typically produces a margin of error of around +/-5.8%.
  So the two are probably roughly within that margin of error or so.


scifact time
 LlamaStackRAGRetriever                            :     0.0225
 MilvusRetriever                                   :     0.0173
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that LlamaStackRAGRetriever generation has a higher score on data of this sort.


fiqa map_cut_10
 LlamaStackRAGRetriever                            :     0.3581
 MilvusRetriever                                   :     0.3581
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 648 questions which typically produces a margin of error of around +/-3.9%.
  So the two are probably roughly within that margin of error or so.


fiqa ndcg_cut_10
 LlamaStackRAGRetriever                            :     0.4411
 MilvusRetriever                                   :     0.4411
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 648 questions which typically produces a margin of error of around +/-3.9%.
  So the two are probably roughly within that margin of error or so.


fiqa time
 LlamaStackRAGRetriever                            :     0.0332
 MilvusRetriever                                   :     0.0303
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that LlamaStackRAGRetriever generation has a higher score on data of this sort.


/Users/bmurdock/beir/beir-venv-310/lib/python3.10/site-packages/scipy/stats/_resampling.py:1492: RuntimeWarning: overflow encountered in scalar power
  n_max = factorial(n_obs_sample)**n_samples
arguana map_cut_10
 LlamaStackRAGRetriever                            :     0.2927
 MilvusRetriever                                   :     0.2927
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 1406 questions which typically produces a margin of error of around +/-2.7%.
  So the two are probably roughly within that margin of error or so.


arguana ndcg_cut_10
 LlamaStackRAGRetriever                            :     0.4251
 MilvusRetriever                                   :     0.4251
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 1406 questions which typically produces a margin of error of around +/-2.7%.
  So the two are probably roughly within that margin of error or so.


arguana time
 LlamaStackRAGRetriever                            :     0.0303
 MilvusRetriever                                   :     0.0239
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that LlamaStackRAGRetriever generation has a higher score on data of this sort.

No significant difference was detected. This is expected because LlamaStackRAGRetriever and MilvusRetriever are intended to do the same thing.  This result is consistent with everything working as intended.
```
