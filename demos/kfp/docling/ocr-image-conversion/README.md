# Docling OCR Image Conversion Pipeline for RAG

  

  

  

This document explains the **Docling OCR (Optical Character Recognition) Image Conversion Pipeline** - a Kubeflow pipeline that processes images using OCR with Docling to extract text and generate embeddings for Retrieval-Augmented Generation (RAG) applications. Pipeline can be run on GPU node and on CPU node without GPU.

  

  

  

  

## 🔄 Pipeline Overview

  

  

  

  

The pipeline transforms images into searchable vector embeddings through the following stages:

  

  

  

  

```mermaid

  

  

  

graph TD

  

  

  

A[Image URLs] --> B[Download Images]

  

  

  

B --> C[Split Images for Parallel Processing]

  

  

  

C --> D[OCR Text Extraction with Docling]

  

  

  

D --> E[Text Chunking]

  

  

  

E --> F[Generate Embeddings]

  

  

  

F --> G[Store in Vector Database]

  

  

  

G --> H[Ready for RAG Queries]

  

  

  

```

  

  

  

  

## 🏗️ Pipeline Components

  

  

  

  

### 1. **Vector Database Registration** (`register_vector_db`)

  

  

  

-  **Purpose**: Sets up the vector database with proper configuration

  

  

  

  

### 2. **Image Import** (`import_test_images`)

  

  

  

-  **Purpose**: Downloads images from remote URLs

  

  

  

### 3. **Image Splitting** (`create_image_splits`)

  

  

  

-  **Purpose**: Distributes images across parallel workers

  

  

  

-  **Process**: Splits images into equal batches for parallel processing

  

  

  

  

### 4. **OCR and Embedding Generation** (`docling_convert_and_ingest_images`)

  

  

  

-  **Purpose**: Main processing component - extracts text and generates embeddings

  

  

  
  
  

  

  

## 🔄 RAG Query Flow

  

  

  

  

1.  **User Query** → Embedding Model → Query Vector

  

  

  

2.  **Vector Search** → Milvus → Similar Chunks

  

  

  

3.  **Context Assembly** → Markdown Content + Metadata

  

  

  

4.  **LLM Generation** → Final Answer with text content from images

  

  

  

  

The pipeline enables rich RAG applications that can answer questions about visual content by leveraging the structured text extracted from images.

  

  

  

  

## 🚀 Getting Started

  

  

### Prerequisites

  

  

- Data Science Project in OpenShift AI with workbench and pipeline server

  - [Data Science Project in OpenShift AI with a configured Workbench](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/html/getting_started)

  -  [Configuring a pipeline server](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.20/html/working_with_data_science_pipelines/managing-data-science-pipelines_ds-pipelines#configuring-a-pipeline-server_ds-pipelines)

- A LlamaStack service with a vector database backend deployed (follow our [official deployment documentation](https://github.com/opendatahub-io/rag/blob/main/DEPLOYMENT.md))

  


  

- GPU-enabled nodes are highly recommended for faster processing.
- You still can use CPU nodes only

### Creating the Pipeline for running on CPU only
```
# Install dependencies for pipeline
cd demos/kfp/docling/ocr-image-conversion
pip3 install -r requirements.txt

# Compile the Kubeflow pipeline for running on CPU only
cd demos/kfp/docling/ocr-image-conversion/conversion-on-cpu
python3 docling_ocr_images_convert_pipeline.py
```


### Creating the Pipeline for running on GPU node
```
# Install dependencies for pipeline
cd demos/kfp/docling/ocr-image-conversion
pip3 install -r requirements.txt

# Compile the Kubeflow pipeline for running with help of GPU
cd demos/kfp/docling/ocr-image-conversion/conversion-on-gpu
python3 docling_ocr_images_convert_pipeline.py
```
  
### Import Kubeflow pipeline to OpenShift AI

  

- Import the compiled YAML to in Pipeline server in your Data Science project in OpenShift AI

    - [Running a data science pipeline generated from Python code](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/html/openshift_ai_tutorial_-_fraud_detection_example/implementing-pipelines#running-a-pipeline-generated-from-python-code)

  

  

- Configure the pipeline parameters as needed

  

  

  

**Pipeline Parameters**

  

  

  

-  `base_url`: URL where image files are hosted

  

  

-  `image_filenames`: Comma-separated list of images to process

  

  

-  `num_workers`: Number of parallel workers (default: 1)

  

  

-  `vector_db_id`: ID for the vector database

  

  

-  `service_url`: URL of the LlamaStack service

  

  

-  `embed_model_id`: Embedding model to use (default: IBM Granite)

  

  

-  `max_tokens`: Maximum tokens per chunk (default: 512)

  

  

-  `use_gpu`: Whether to use GPU for processing (default: true)

  

  

### Query RAG Agent in your Workbench within a Data Science project on OpenShift AI
1. Open your Workbench
2. Clone the rag repo and use main branch
	- Use this link `https://github.com/opendatahub-io/rag.git` for cloning the repo
	- [Collaborating on Jupyter notebooks by using Git](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/html/working_with_connected_applications/using_basic_workbenches#collaborating-on-jupyter-notebooks-by-using-git_connected-apps)
			         
3. Install dependencies for Jupyter Notebook with RAG Agent
```
cd demos/kfp/docling/ocr-image-conversion/rag-agent
pip3 install -r requirements.txt 
```
4. Follow the instructions in the corresponding RAG Jupyter Notebook `ocr_images_rag_agent.ipynb` to query the content ingested by the pipeline.
```
cd demos/kfp/docling/ocr-image-conversion/rag-agent
```