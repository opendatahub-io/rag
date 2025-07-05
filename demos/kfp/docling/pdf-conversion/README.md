# 🚀 PDF Ingestion and RAG Indexing with Docling, LlamaStack, and Milvus

## Overview

This pipeline converts your PDF documents to Markdown format using Docling, chunks the content, generates embeddings with a SentenceTransformer model, and inserts those embeddings into your vector database via Llama Stack for efficient vector search. It’s designed to run on Kubeflow Pipelines with GPU acceleration support.

## Prerequisites

- OpenShift AI with a data science project that includes a configured pipeline server
- [LlamaStack Operator](https://github.com/opendatahub-io/llama-stack-k8s-operator) installed
- LlamaStackDistribution custom resource [configured](../../../stack/README.md)

## Resource Requirements

- CPU: 500m-4 cores
- Memory: 2-6 Gi
- GPU: 1 NVIDIA GPU (when `use_gpu` is enabled)

## Pipeline Components

### 0. Register vector DB (`register_vector_db`)

- Registers a vector database in LlamaStack using the provided embedding model.

### 1. Import PDFs (`import_test_pdfs`)

- Downloads PDF documents from a given base URL.
- Copies the downloaded PDFs to the pipeline workspace.

### 2. Create PDF Splits (`create_pdf_splits`)

- Divides PDFs into batches for parallel processing
- Configurable number of splits based on available workers

### 3. Docling Convert and Ingest data into Llama Stack's Vector Store (`docling_convert_and_ingest`)

- Converts PDFs to Markdown using Docling
- Generates embeddings using sentence transformers
- Stores embeddings in Milvus vector database
- Supports GPU acceleration for faster processing

## Import and run the KubeFlow Pipeline

Key pipeline parameters:

- `base_url`: The base web URL where the source PDF files are located.
- `pdf_filenames`: A comma-separated string of PDF filenames to download from the base_url.
- `num_workers`: Number of parallel workers
- `vector_db_id`: Milvus vector database ID
- `service_url`: Milvus service URL
- `embed_model_id`: Embedding model to use
- `max_tokens`: Maximum tokens per chunk
- `use_gpu`: Enable/disable GPU acceleration

## Prompt the LLM

- Once your documents are embedded and indexed, you can query them using a Retrieval-Augmented Generation (RAG) workflow.
- You have two options to interact with your indexed content:
  - ChatBot UI
  - Example Notebook [docling_rag.ipynb](docling_rag.ipynb)
