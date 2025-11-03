# Copyright 2025 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: PLC0415,UP007,UP035,UP006,E712
# SPDX-License-Identifier: Apache-2.0
from kfp import compiler, dsl
from kfp.kubernetes import add_node_selector_json, add_toleration_json

# Workbench Runtime Image: Pytorch with CUDA and Python 3.12 (UBI 9)
# The images for each release can be found in
# https://github.com/red-hat-data-services/rhoai-disconnected-install-helper/blob/main/rhoai-2.23.md
PYTORCH_CUDA_IMAGE = "quay.io/modh/odh-pipeline-runtime-pytorch-cuda-py312-ubi9@sha256:72ff2381e5cb24d6f549534cb74309ed30e92c1ca80214669adb78ad30c5ae12"


@dsl.component(
    base_image=PYTORCH_CUDA_IMAGE,
    packages_to_install=["llama-stack-client", "fire", "requests"],
)
def register_vector_store_and_files(
    service_url: str,
    vector_store_name: str,
    embedding_model_id: str,
    max_tokens: int,
    chunk_overlap_tokens: int,
    base_url: str,
    pdf_filenames: str,
):
    import io
    import requests
    from llama_stack_client import LlamaStackClient

    client = LlamaStackClient(base_url=service_url)

    # Upload all files first and collect file_ids
    file_ids = []
    for filename in pdf_filenames.split(","):
        source = f"{base_url}/{filename.strip()}"
        print("Downloading and uploading document:", source)

        try:
            # Download the docs from URL
            response = requests.get(source)
            response.raise_for_status()  # Raise an exception for bad status codes

            file_content = io.BytesIO(response.content)
            file_basename = source.split("/")[-1]

            # Upload file to storage
            file = client.files.create(
                file=(file_basename, file_content, "application/pdf"),
                purpose="assistants",
            )
            file_ids.append(file.id)
            print(f"Successfully uploaded {file_basename} (file_id: {file.id})")

        except Exception as e:
            print(f"ERROR: Failed to upload {filename.strip()}: {str(e)}")
            raise

    print(f"Successfully uploaded {len(file_ids)} files: {file_ids}")

    models = client.models.list()
    matching_model = next(
        (m for m in models if m.provider_resource_id == embedding_model_id), None
    )

    if not matching_model:
        raise ValueError(
            f"Model with ID '{embedding_model_id}' not found on LlamaStack server."
        )

    if matching_model.api_model_type != "embedding":
        raise ValueError(f"Model '{embedding_model_id}' is not an embedding model")

    embedding_dimension = matching_model.metadata["embedding_dimension"]

    # Create vector store from uploaded files
    try:
        vector_store = client.vector_stores.create(
            name=vector_store_name,
            file_ids=file_ids,
            chunking_strategy={
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": max_tokens,
                    "chunk_overlap_tokens": chunk_overlap_tokens,
                },
            },
            extra_body={
                "embedding_model": embedding_model_id,
                "embedding_dimension": embedding_dimension,
                "provider_id": "milvus",
            },
        )
        print(
            f"Successfully created vector store '{vector_store_name}' with ID: {vector_store.id}"
        )
        print(f"Vector store details: {vector_store}")
    except Exception as e:
        print(f"ERROR: Failed to create vector store '{vector_store_name}': {str(e)}")
        raise


@dsl.pipeline()
def vector_store_files_pipeline(
    base_url: str = "https://raw.githubusercontent.com/ChristianZaccaria/redbank-kb/main",
    pdf_filenames: str = "redbankfinancial_about.pdf, redbankfinancial_faq.pdf",
    vector_store_name: str = "redbank-kb-vector-store",
    service_url: str = "http://lsd-llama-milvus-service:8321",
    embedding_model_id: str = "ibm-granite/granite-embedding-125m-english",
    max_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    use_gpu: bool = True,
):
    """
    Creates a vector store with embeddings from PDF files from a GitHub source.
    :param base_url: Base URL to fetch PDF files from
    :param pdf_filenames: Comma-separated list of PDF filenames to download and convert
    :param vector_store_name: Name of the vector store to store embeddings
    :param service_url: URL of the Milvus service
    :param embedding_model_id: Model ID for embedding generation
    :param max_tokens: Maximum number of tokens per chunk
    :param chunk_overlap_tokens: Number of overlapping tokens between chunks
    :param use_gpu: Enable GPU usage for embedding generation
    :return:
    """

    with dsl.If(use_gpu == True):
        register_task = register_vector_store_and_files(
            service_url=service_url,
            vector_store_name=vector_store_name,
            embedding_model_id=embedding_model_id,
            max_tokens=max_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            base_url=base_url,
            pdf_filenames=pdf_filenames,
        )
        register_task.set_caching_options(False)
        register_task.set_cpu_request("500m")
        register_task.set_cpu_limit("4")
        register_task.set_memory_request("2Gi")
        register_task.set_memory_limit("6Gi")
        register_task.set_accelerator_type("nvidia.com/gpu")
        register_task.set_accelerator_limit(1)
        add_toleration_json(
            register_task,
            [
                {
                    "effect": "NoSchedule",
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                }
            ],
        )
        add_node_selector_json(register_task, {})

    with dsl.Else():
        register_task = register_vector_store_and_files(
            service_url=service_url,
            vector_store_name=vector_store_name,
            embedding_model_id=embedding_model_id,
            max_tokens=max_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            base_url=base_url,
            pdf_filenames=pdf_filenames,
        )
        register_task.set_caching_options(False)
        register_task.set_cpu_request("500m")
        register_task.set_cpu_limit("4")
        register_task.set_memory_request("2Gi")
        register_task.set_memory_limit("6Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(
        vector_store_files_pipeline,
        package_path=__file__.replace(".py", "_compiled.yaml"),
    )
