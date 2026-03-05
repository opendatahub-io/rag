# 🚀 RAGAS Evaluation Demo with SDG Hub RAG Flow for Real World RAG  Systems

> [!WARNING]
> **This is a demo/development setup only - NOT for production use!**
>
> This demo uses hardcoded credentials and insecure configurations for ease of setup:
> - MinIO default credentials (`minio`/`password`)
> - PostgreSQL default credentials (`llamastack`/`llamastack`)
> - OAuth disabled on Data Science Pipelines
> - No TLS encryption for internal communication
>
> For production deployments, use proper secret management, enable OAuth/RBAC, use managed services, and enable TLS.

## Overview

This demo showcases how to generate synthetic RAG evaluation datasets using the SDG Hub RAG Flow and evaluate them using the RAGAS Llama Stack Eval Provider developed by the Trusty AI team. The demo demonstrates:

1. **Synthetic Dataset Generation**: Using SDG Hub to create questions with ground truth context from documents
2. **Real World Scenario with Document Ingestion and LLM Answers**: An LLM is tasked with answering the synthetically generated questions by searching its own Vector Store for data.
3. **RAGAS Evaluation**: Using the RAGAS provider to evaluate RAG systems with metrics like faithfulness, answer relevancy, context precision, and context recall

The demo includes notebooks for dataset generation and evaluation, running on Red Hat OpenShift AI.

This guide assumes RHOAI 3.4+ is installed on an OpenShift 4.19.9+ cluster.

> Note: This demo was tested using the default KServe behavior on OpenShift AI (`Headless` RawDeployment). If you are using `Headed` mode, change the `VLLM_URL` port to `80` in the [llama-stack-distribution.yaml](deployment-yamls/llama-stack-distribution.yaml).

## Key Features

This demo includes a **self-contained deployment** with:
- **MinIO** for S3-compatible object storage (no external AWS account needed)
- **PostgreSQL** for llama-stack data storage
- **Data Science Pipelines (DSPA)** deployed automatically via manifests
- **Network policies** for secure pod-to-pod communication

---

## Table of Contents

- [Initial Setup](#initial-setup)
  - [Instructions to get GPUs on OpenShift](#instructions-to-get-gpus-on-openshift)
  - [Update RHOAI DSC](#update-rhoai-dsc)
  - [Create a Workbench](#create-a-workbench)
- [Deploy AI Models and Llama Stack Server](#deploy-ai-models-and-llama-stack-server)
- [Run the Evaluation Flow](#run-the-evaluation-flow)
  - [Generate Synthetic Dataset](#generate-synthetic-dataset)
  - [Run RAGAS Evaluation](#run-ragas-evaluation)

---

## Initial Setup

### Instructions to get GPUs on OpenShift

#### Get GPU Worker Nodes

**Steps:**

1. Go to the Openshift cluster console.
2. Under `<your-cluster>` → Machine pools, click "Add machine pool".
3. Add a name, and in "Compute node instance type" scroll all way down and search for `g5.2xlarge`. This demo has been tested with 2 `g5.2xlarge` nodes (A10g NVIDIA GPU), but should work with other similar NVIDIA GPU instances.
4. Click on Add machine pool.

---

#### Install the GPU Operators

**Steps:**

1. Go to the openshift dashboard.
2. In OperatorHub install the following operators:
   - **Node Feature Discovery Operator** - install with default settings
     - Create Node Feature Discovery CR, the defaults are fine
     - Several pods will start in the `openshift-nfd` (default) namespace. Once all these are up, the nodes will be labeled with a lot of feature flags. At which point you can proceed.
   - **NVIDIA GPU Operator** - install with default settings
     - Create GPU ClusterPolicy CR, the defaults are fine. This will create several pods in the nvidia GPU namespace, they can take a while to come up because they compile the driver. Once they are up, scheduler should have allocatable GPUs.

---

### Update RHOAI DSC

#### Enable the Llama Stack K8s Operator

**Steps:**

1. In the RHOAI DSC custom resource, enable `llamastackoperator`:

```yaml
llamastackoperator:
  managementState: Managed
```

**Verification:**

See the `llama-stack-k8s-operator` running in the `redhat-ods-applications` namespace.

---

### Create a Workbench

**Steps:**

1. In the OpenShift console, click the grid icon in the top-right corner, then select Red Hat OpenShift AI.
2. Click on the Create project button.
3. Name the project `ragas-evaluation`. This creates the `ragas-evaluation` namespace used by all subsequent deployments.
4. Click on Create a workbench, then configure:
   - **Name:** `ragas-evaluation-workbench`
   - **Image selection:** Jupyter | Minimal | CPU | Python 3.12
   - **Version selection:** 2025.2
5. Leave the remaining settings as defaults, then click Create workbench.

**Verification:**

After the workbench has initialized, confirm that the workbench status shows as Running.

---

Once you've completed these steps, continue with the [Deploy AI Models and Llama Stack Server](#deploy-ai-models-and-llama-stack-server) section below.

---

## Deploy AI Models and Llama Stack Server

> **Note:** The Qwen3-14B-AWQ model uses 1 GPU. This demo has been tested with `g5.2xlarge` nodes (A10g NVIDIA GPU).

This deployment automatically sets up the following components:
- **MinIO** - S3-compatible object storage for pipelines and results
- **PostgreSQL** - Database for llama-stack
- **DSPA** - Data Science Pipelines Application (with OAuth disabled for simplicity)
- **Network Policy** - Allows llama-stack to communicate with DSPA
- **Qwen3-14B-AWQ** - Inference model
- **GPT-OSS-20b** - Inference model
- **Llama Stack Distribution** - Llama Stack server with RAGAS evaluation provider

**Deploy:**

Run in your terminal:
```bash
make deploy-all
```

This will:
1. Create the `ragas-evaluation` namespace
2. Deploy configuration secrets and ConfigMaps
3. Deploy MinIO for S3-compatible storage
4. Deploy PostgreSQL for llama-stack database
5. Deploy DSPA (Data Science Pipelines Application)
6. Deploy the Qwen3-14B-AWQ inference model
7. Deploy the GPT-OSS inference model
8. Deploy the Llama Stack Distribution server

**Verification:**

Wait until all pods are fully running in the `ragas-evaluation` namespace. You can check the status with:

```bash
oc get pods -n ragas-evaluation
```

You should see the following pods:
- `minio-*` (S3-compatible object storage)
- `postgres-*` (llama-stack database)
- `ds-pipeline-*`, `mariadb-*` (Data Science Pipelines)
- `qwen3-14b-awq-predictor-*` (inference model)
- `redhataigpt-oss-20b-predictor-*` (inference model)
- `lsd-ragas-example-*` (Llama Stack Distribution)

**Cleanup:**

To tear down all deployed resources:

```bash
make delete-all
```

## Run the Evaluation Flow

### Generate Synthetic Dataset

**Steps:**

1. Go to the Red Hat OpenShift AI dashboard, and go into your workbench. This will open your JupyterNotebook environment.
2. Upload the `notebooks` directory to your JupyterNotebook environment.
3. Open the `1.dataset_generation.ipynb` file.
4. Follow the notebook steps to:
   - Prepare your input dataset (documents with outlines)
   - Configure the SDG Hub RAG Flow
   - Generate synthetic question-answer pairs with ground truth context
   - Post-process the results for evaluation

**Note:** The notebook includes an example using the IBM Annual Report 2024 PDF (`ibm-annual-report-2024.pdf`). You can use your own documents by modifying the input dataset preparation section.

### Run RAG Inference

**Steps:**

1. In the same JupyterNotebook environment, open the `2.rag_inference.ipynb` file.
2. Follow the notebook steps to:
   - Connect to Llama Stack and discover available inference and embedding models
   - Upload the same source PDF used in notebook 1
   - Create a vector store (Milvus-backed) with chunked embeddings
   - Load synthetic questions from `rag_evaluation_dataset.jsonl`
   - Run each question through the RAG pipeline using `file_search`
   - Save RAG answers and retrieved contexts to `rag_inference_dataset.jsonl`

> [!WARNING]
> **By default trust_remote_code is set to False for sentence transformers and cannot be changed**
> Only Embedding models that work with trust_remote_code=False will work with this demo e.g. `ibm-granite/granite-embedding-125m-english`

**Note:** Ensure `rag_evaluation_dataset.jsonl` exists (from notebook 1) and the source PDF (e.g. `ibm-annual-report-2024.pdf`) is in the current directory. Optionally run the cleanup cell to delete the vector store when finished.

### Run RAGAS Evaluation

**Steps:**

1. In the same JupyterNotebook environment, open the `3.ragas-evaluation.ipynb` file.
2. Follow the notebook steps to:
   - Load the RAG inference dataset from `rag_inference_dataset.jsonl`
   - Register the dataset with Llama Stack (Datasets API)
   - Configure RAGAS evaluation metrics (e.g. answer relevancy, context precision, faithfulness, context recall)
   - Run evaluation using the RAGAS provider (remote mode via Kubeflow Pipelines)
   - Display and analyze the results (per-question scores and aggregate metrics)

**Note:** Complete notebooks 1 and 2 first so that `rag_inference_dataset.jsonl` exists. You can enable or disable individual RAGAS metrics in the benchmark registration step.

---
