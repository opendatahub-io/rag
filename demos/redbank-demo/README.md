# 🚀 Red Bank Demo

## Overview

This demo showcases how to build an end-to-end AI assistant on Red Hat OpenShift AI (RHOAI) using Llama Stack and KServe.

It covers document ingestion and vectorization, model registration, Retrieval-Augmented Generation (RAG), speech-to-text (STT) with Whisper, and text-to-speech (TTS) with Kokoro.

By the end, you'll have a fully working “Red Bank Assistant” capable of understanding questions, retrieving answers from internal documents, and responding with natural speech.

This guide assumes RHOAI 3.0+ is installed on an OpenShift 4.19.9+ cluster.

---

## Table of Contents

- [Instructions to get GPUs on OpenShift](#instructions-to-get-gpus-on-openshift)
  - [Get a GPU Worker Node](#get-a-gpu-worker-node)
  - [Install the GPU Operators](#install-the-gpu-operators)
  - [Accelerator Migration](#accelerator-migration)
- [Update RHOAI DSC](#update-rhoai-dsc)
  - [Enable the Llama Stack K8s Operator](#enable-the-llama-stack-k8s-operator)
  - [Update KServe](#update-kserve)
- [Create and Configure a Data Science Project](#create-and-configure-a-data-science-project)
- [Create Data Science Pipeline](#create-data-science-pipeline)
  - [Create AWS S3 Bucket](#create-aws-s3-bucket)
  - [Create Data Science Pipeline Server](#create-data-science-pipeline-server)
  - [Create a Pipeline](#create-a-pipeline)
- [Deploy STT, TTS, and Qwen-2.5:7b AI models](#deploy-stt-tts-and-qwen-257b-ai-models)
- [Run the Data Science Pipeline to create Red Bank knowledge base](#run-the-data-science-pipeline-to-create-red-bank-knowledge-base)
- [See RAG and MCP tooling in action with your TTS, STT, and Qwen-2.5:7b AI models](#see-rag-and-mcp-tooling-in-action-with-your-tts-stt-and-qwen-257b-ai-models)

---

## Instructions to get GPUs on OpenShift

### Get GPU Worker Nodes

**Steps:**

1. Go to the Openshift cluster console.
2. Under `<your-cluster>` → Machine pools, click "Add machine pool".
3. Add a name, and in "Compute node instance type" scroll all way down and search for `g5.2xlarge`.
4. Click on Add machine pool.
5. Repeat the steps above to provision 2 `g4dn.xlarge` GPUs. These will be used for the STT Whisper model and TTS Kokoro model during this demo.

---

### Install the GPU Operators

**Steps:**

1. Go to the openshift dashboard.
2. In OperatorHub install the following operators:
   - **Node Feature Discovery Operator**
     - Create Node Feature Discovery CR, the defaults are fine
     - Several pods will start in the `openshift-nfd` (default) namespace. Once all these are up, the nodes will be labeled with a lot of feature flags. At which point you can proceed.
   - **NVIDIA GPU Operator**
     - Create GPU ClusterPolicy CR. This will create several pods in the nvidia GPU namespace, they can take a while to come up because they compile the driver. Once they are up, scheduler should have allocatable GPUs.

---

### Accelerator Migration

If you already have RHOAI already deployed, you need to force the migration.

**Steps:**

1. Go to the `redhat-ods-applications` namespace.
2. Go to the Configmaps.
3. Delete the one called `migration-gpu-status`.
4. Now go to Deployments.
5. Click into `rhods-dashboard`.
6. Go to the replicasets.
7. Delete the the replicaset to force a restart of the pods.
8. No go to Search > Resources > AcceleratorProfiles.

**Verification:**

You should see a resource under the `redhat-ods-applications` namespace. If not, just install the RHOAI operator and you should be ready to go.

---

## Update RHOAI DSC

### Enable the Llama Stack K8s Operator

**Steps:**

1. In the RHOAI DSC custom resource, enable `llamastackoperator`:

```yaml
llamastackoperator:
  managementState: Managed
```

**Verification:**

See the `llama-stack-k8s-operator` running in the `redhat-ods-applications` namespace.

---

### Update KServe

**Steps:**

1. In the RHOAI DSC custom resource, under the `kserve` section, set `rawDeploymentServiceConfig` to `Headed`:

```yaml
kserve:
  managementState: Managed
  nim:
    managementState: Managed
  rawDeploymentServiceConfig: Headed
```

**Verification:**

See the `kserve-controller-manager` running in the `redhat-ods-applications` namespace.

---

## Create and Configure a Data Science Project

**Steps:**

1. In the OpenShift console, click the grid icon in the top-right corner, then select Red Hat OpenShift AI.
2. Click on the Create project button.
3. Name the project `redbank-demo`.
4. Click on Create a workbench, then configure:
   - **Name:** `redbank-workbench`
   - **Image selection:** Jupyter | Minimal | CPU | Python 3.12
   - **Version selection:** 2025.2
5. Leave the remaining settings as defaults, then click Create workbench.

**Verification:**

After the workbench has initialized, confirm that the workbench status shows as Running.

---

## Create Data Science Pipeline

To create a Data Science Pipeline, we need to set up an AWS S3 bucket to store pipeline artifacts.

### Create AWS S3 Bucket

**Steps:**

1. Log-in to AWS Console and go to the S3 Service.
2. Click on Create bucket.
3. Enter a bucket name, such as `redbank-s3-bucket`.
4. Leave the remaining settings as defaults, then click on Create bucket.
5. Search for and select your newly created S3 bucket, then open the Properties tab to find and note the AWS Region (e.g., `us-east-1`).

---

### Create Data Science Pipeline Server

**Steps:**

1. In the Red Hat OpenShift AI dashboard, navigate to the Develop & train dropdown on the left sidebar and select Pipelines, then select Pipeline definitions. Choose your project (`redbank-demo`), then click Configure pipeline server.
2. Configure the pipeline server:
   - **Access key:** `<aws-access-key>`
   - **Secret key:** `<aws-secret-key>`
   - **Endpoint:** `s3.<aws-region>.amazonaws.com`
   - **Region:** `<aws-region>`
   - **Bucket:** `redbank-s3-bucket`
3. Click on Configure pipeline server.
4. Wait until the pipeline server has initialized.

---

### Create a Pipeline

**Steps:**

1. Click on Import pipeline.
2. Name the pipeline `redbank-kb-pipeline`.
3. Upload the `vector_store_files_pipeline_compiled.yaml` file.
4. Click on Import pipeline.

---

## Deploy STT, TTS, and Qwen-2.5:7b AI models + MCP Server and PostgreSQL

> **Note:** The Qwen-2.5:7b model runs on 1 GPU. We are using a `g5.2xlarge` node (A10g NVIDIA GPU), and the Whisper (STT) model
and Kokoro (TTS) model each run on 1 `g4dn.xlarge` node (NVIDIA T4 GPU)

**Steps:**

1. Run in your terminal:
```bash
  make deploy-all
```
**Verification:**
Wait until all pods are fully running in the `redbank-demo` namespace.

---

## Run the Data Science Pipeline to create Red Bank knowledge base

**Steps:**

1. In the Red Hat OpenShift AI dashboard, navigate to the Data science pipelines dropdown on the left sidebar and select Pipelines.
2. Select `redbank-kb-pipeline`.
3. On the top-right corner click on the Actions dropdown, and select Create run.
4. Configure the pipeline run, and the rest can be left as defaults:
   - **Name:** `redbank-kb-run`
   - **use_gpu:** False (unless you have additional NVIDIA GPUs available)
5. Click on Create run.
6. Wait for the pipeline run to complete.

---

## See RAG and MCP tooling in action with your TTS, STT, and Qwen-2.5:7b AI models

**Steps:**
1. Go to the Red Hat OpenShift AI dashboard, and go into your workbench. This will open your JupyterNotebook environment.
2. Upload the `notebook` directory to your JupyterNotebook environment.
3. Open the `redbank_notebook.ipynb` file and run through all cells to test out all your models and RAG.
4. RAG retrieval will be performed, and the LLM should accurately answer the question based on our documents.
