# 🚀 PDF Ingestion and RAG Indexing with Docling, LlamaStack, and Milvus

## Overview

This pipeline converts your PDF documents to Markdown format using Docling, chunks the content, generates embeddings with a SentenceTransformer model, and inserts those embeddings into your vector database via Llama Stack for efficient vector search. It’s designed to run on Kubeflow Pipelines with GPU acceleration support.

## Prerequisites
- A Kubeflow Pipelines environment
- LlamaStack Operator installed
- LlamaStackDistribution custom resource configured

## Pipeline Components

### 1. Import PDFs (`import_test_pdfs`)
- Clones a Git repository containing PDF documents
- Copies PDFs to the pipeline workspace

### 2. Create PDF Splits (`create_pdf_splits`)
- Divides PDFs into batches for parallel processing
- Configurable number of splits based on available workers

### 3. Docling Convert (`docling_convert`)
- Converts PDFs to Markdown using Docling
- Generates embeddings using sentence transformers
- Stores embeddings in Milvus vector database
- Supports GPU acceleration for faster processing

## Configuration

Key pipeline parameters:
- `input_docs_git_repo`: Git repository URL containing PDFs
- `input_docs_git_branch`: Git branch to use
- `input_docs_git_folder`: Folder containing PDFs in the repository
- `num_workers`: Number of parallel workers
- `vector_db_id`: Milvus vector database ID
- `service_url`: Milvus service URL
- `embed_model_id`: Embedding model to use
- `max_tokens`: Maximum tokens per chunk
- `use_gpu`: Enable/disable GPU acceleration

## Resource Requirements

- CPU: 500m-4 cores
- Memory: 2-4 Gi
- GPU: 1 NVIDIA GPU (when enabled)
