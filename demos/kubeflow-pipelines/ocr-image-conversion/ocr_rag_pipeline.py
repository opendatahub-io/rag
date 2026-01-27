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

PYTHON_BASE_IMAGE = "registry.redhat.io/ubi9/python-312@sha256:e80ff3673c95b91f0dafdbe97afb261eab8244d7fd8b47e20ffcbcfee27fb168"


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=[
        "llama-stack-client==0.4.2",
        "fire",
        "requests",
        "pytesseract",
        "Pillow",
    ],
)
def register_vector_store_and_files(
    service_url: str,
    vector_store_name: str,
    embedding_model_id: str,
    max_tokens: int,
    chunk_overlap_tokens: int,
    base_url: str,
    image_filenames: str,
):
    import io
    import os
    import tempfile
    import requests
    import pytesseract
    from PIL import Image
    from llama_stack_client import LlamaStackClient

    def download_and_install_tesseract():
        import subprocess
        import pathlib

        try:
            subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
            print("Tesseract OCR is already installed")
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        print("Downloading Tesseract OCR...")

        import urllib.request

        install_dir = pathlib.Path("/tmp/tesseract")
        install_dir.mkdir(exist_ok=True)
        bin_dir = pathlib.Path("/tmp/bin")
        bin_dir.mkdir(exist_ok=True)

        # Download Tesseract AppImage (self-contained with tessdata)
        url = "https://github.com/AlexanderP/tesseract-appimage/releases/download/v5.5.2/tesseract-5.5.2-x86_64.AppImage"
        appimage = install_dir / "tesseract.AppImage"
        urllib.request.urlretrieve(url, appimage)
        os.chmod(appimage, 0o755)

        subprocess.run(
            [str(appimage), "--appimage-extract"],
            cwd=str(install_dir),
            check=True,
            capture_output=True,
        )

        # Create wrapper script to call extracted AppRun
        wrapper = bin_dir / "tesseract"
        wrapper.write_text(
            f'#!/bin/sh\nexec "{install_dir}/squashfs-root/AppRun" "$@"\n'
        )
        os.chmod(wrapper, 0o755)

        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        print("Tesseract OCR installed successfully")

    download_and_install_tesseract()

    client = LlamaStackClient(base_url=service_url)

    # Process images and upload OCR text
    file_ids = []
    for filename in image_filenames.split(","):
        source = f"{base_url}/{filename.strip()}"
        print(f"Downloading and processing image: {source}")

        try:
            # Download the image file
            response = requests.get(source)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_image:
                tmp_image.write(response.content)
                tmp_image_path = tmp_image.name

            try:
                # Perform OCR on the image
                print(f"Performing OCR on {filename.strip()}...")
                image = Image.open(tmp_image_path)
                ocr_text = pytesseract.image_to_string(image)
                print(f"OCR complete: {len(ocr_text)} characters extracted")

                # Upload OCR text as text file
                file_basename = filename.strip().rsplit(".", 1)[0] + ".txt"
                text_content = io.BytesIO(ocr_text.encode("utf-8"))

                file = client.files.create(
                    file=(file_basename, text_content, "text/plain"),
                    purpose="assistants",
                )
                file_ids.append(file.id)
                print(
                    f"Successfully uploaded OCR text {file_basename} (file_id: {file.id})"
                )

            finally:
                # Clean up temp file
                os.unlink(tmp_image_path)

        except Exception as e:
            print(f"ERROR: Failed to process {filename.strip()}: {str(e)}")
            raise

    print(f"Successfully processed and uploaded {len(file_ids)} OCR texts: {file_ids}")

    models = client.models.list()
    matching_model = next((m for m in models if m.id == embedding_model_id), None)

    if not matching_model:
        available = [m.id for m in models]
        raise ValueError(
            f"Model '{embedding_model_id}' not found. Available: {available}"
        )

    model_type = (
        matching_model.custom_metadata.get("model_type")
        if matching_model.custom_metadata
        else None
    )
    if model_type != "embedding":
        raise ValueError(
            f"Model '{embedding_model_id}' is not an embedding model (type={model_type})"
        )

    embedding_dimension = int(
        float(matching_model.custom_metadata.get("embedding_dimension"))
    )

    # Warm up the embedding model
    client.embeddings.create(
        model=embedding_model_id,
        input="warmup",
    )

    # Create empty vector store first, before inserting files.
    # Purpose: Depending on the size and number of files, attempting to create the vector store
    # and add files in a single step may lead to timeouts.
    try:
        vector_store = client.vector_stores.create(
            name=vector_store_name,
            file_ids=[],
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
    except Exception as e:
        print(f"ERROR: Failed to create vector store '{vector_store_name}': {str(e)}")
        raise

    # Add files to vector store
    try:
        for file_id in file_ids:
            print(f"Adding file_id '{file_id}' to vector store '{vector_store_name}'")
            client.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=file_id,
            )
        vector_store = client.vector_stores.retrieve(vector_store.id)
        print(f"Vector store details: {vector_store}")
    except Exception as e:
        print(f"WARNING: Some files failed to be added to vector store: {str(e)}")


@dsl.pipeline()
def vector_store_files_pipeline(
    base_url: str = "https://raw.githubusercontent.com/opendatahub-io/rag/main/demos/testing-data/images",
    image_filenames: str = "RAG_flow_diagram.png, RAG_key_market_usecases.png",
    vector_store_name: str = "ocr-vector-store",
    service_url: str = "http://lsd-milvus-service:8321",
    embedding_model_id: str = "sentence-transformers/ibm-granite/granite-embedding-125m-english",
    max_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    use_gpu: bool = False,
) -> None:
    """
    Converts images to text using OCR and generates embeddings
    :param base_url: Base URL to fetch image files from
    :param image_filenames: Comma-separated list of image filenames to download and convert
    :param vector_store_name: Name of the vector store to store embeddings
    :param service_url: URL of the LlamaStack service
    :param embedding_model_id: Model ID for embedding generation
    :param max_tokens: Maximum number of tokens per chunk
    :param chunk_overlap_tokens: Number of overlapping tokens between chunks
    :param use_gpu: boolean to enable/disable gpu
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
            image_filenames=image_filenames,
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
            image_filenames=image_filenames,
        )
        register_task.set_caching_options(False)
        register_task.set_cpu_request("500m")
        register_task.set_cpu_limit("4")
        register_task.set_memory_request("2Gi")
        register_task.set_memory_limit("6Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=vector_store_files_pipeline,
        package_path=__file__.replace(".py", "_compiled.yaml"),
    )
