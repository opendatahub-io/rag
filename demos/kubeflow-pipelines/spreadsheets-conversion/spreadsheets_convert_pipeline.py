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
from typing import List
import logging

from kfp import compiler, dsl
from kfp.kubernetes import add_node_selector_json, add_toleration_json

PYTHON_BASE_IMAGE = "registry.redhat.io/ubi9/python-312@sha256:e80ff3673c95b91f0dafdbe97afb261eab8244d7fd8b47e20ffcbcfee27fb168"
LLAMA_STACK_CLIENT_VERSION = "0.4.2"

_log = logging.getLogger(__name__)


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=[
        f"llama-stack-client=={LLAMA_STACK_CLIENT_VERSION}",
        "fire",
        "requests",
    ],
)
def create_vector_store(
    service_url: str,
    vector_store_name: str,
    embedding_model_id: str,
    max_tokens: int,
    chunk_overlap_tokens: int,
) -> str:
    """Create an empty vector store for file_search (Responses API). Returns vector_store.id."""
    from llama_stack_client import LlamaStackClient

    client = LlamaStackClient(base_url=service_url)

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
    print(f"Created vector store '{vector_store_name}' (id={vector_store.id})")
    return vector_store.id


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["requests"],
)
def import_spreadsheet_files(
    base_url: str,
    spreadsheet_filenames: str,
    output_path: dsl.OutputPath("input-spreadsheets"),
):
    import os
    import requests
    import shutil

    os.makedirs(output_path, exist_ok=True)
    filenames = [f.strip() for f in spreadsheet_filenames.split(",") if f.strip()]

    for filename in filenames:
        url = f"{base_url.rstrip('/')}/{filename}"
        file_path = os.path.join(output_path, filename)

        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
            print(f"Downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}: {e}, skipping.")


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
)
def create_spreadsheet_splits(
    input_path: dsl.InputPath("input-spreadsheets"),
    num_splits: int,
) -> List[List[str]]:
    import pathlib

    # Split our entire directory of spreadsheet files into n batches, where n == num_splits
    # Support common formats
    spreadsheet_extensions = ["*.csv", "*.xlsx", "*.xls", "*.xlsm"]
    all_spreadsheets = []

    input_dir = pathlib.Path(input_path)
    for ext in spreadsheet_extensions:
        all_spreadsheets.extend([path.name for path in input_dir.glob(ext)])

    splits = [
        batch
        for batch in (all_spreadsheets[i::num_splits] for i in range(num_splits))
        if batch
    ]
    return splits or [[]]


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=[
        f"llama-stack-client=={LLAMA_STACK_CLIENT_VERSION}",
        "pandas>=2.3.0",
        "openpyxl>=3.1.5",
    ],
)
def convert_and_upload_spreadsheets(
    input_path: dsl.InputPath("input-spreadsheets"),
    spreadsheet_split: List[str],
    service_url: str,
    vector_store_id: str,
):
    import io
    import logging
    import pathlib
    import shutil
    import tempfile

    import pandas as pd
    from llama_stack_client import LlamaStackClient

    _log = logging.getLogger(__name__)

    local_processing_dir = pathlib.Path(tempfile.mkdtemp(prefix="spreadsheets-"))
    _log.info(f"Local processing directory: {local_processing_dir}")

    def convert_excel_to_csv(
        input_spreadsheet_files: List[pathlib.Path], output_path: pathlib.Path
    ) -> List[pathlib.Path]:
        processed_csv_files = []

        for file_path in input_spreadsheet_files:
            if not file_path.exists():
                _log.info(f"Skipping missing file: {file_path}")
                continue

            if file_path.suffix.lower() == ".csv":
                new_path = output_path / file_path.name
                try:
                    df = pd.read_csv(file_path, compression="infer", engine="python")
                    _log.info(f"Read {file_path.name} as a standard CSV.")

                except (UnicodeDecodeError, EOFError):
                    _log.warning(
                        f"Standard read failed for {file_path.name}. Attempting gzip decompression."
                    )
                    try:
                        # Second, try reading it again, but force gzip decompression.
                        df = pd.read_csv(file_path, compression="gzip", engine="python")
                        _log.info(
                            f"Successfully read {file_path.name} with forced gzip."
                        )
                    except Exception as e:
                        _log.error(
                            f"Could not read {file_path.name} with any method. Error: {e}"
                        )
                        continue

                df.to_csv(new_path, index=False)
                processed_csv_files.append(new_path)

            elif file_path.suffix.lower() in [".xlsx", ".xls", ".xlsm"]:
                _log.info(f"Converting {file_path.name} to CSV format...")
                try:
                    excel_sheets = pd.read_excel(file_path, sheet_name=None)
                    for sheet_name, df in excel_sheets.items():
                        new_csv_filename = f"{file_path.stem}_{sheet_name}.csv"
                        new_csv_path = output_path / new_csv_filename
                        df.to_csv(new_csv_path, index=False, header=True)
                        processed_csv_files.append(new_csv_path)
                        _log.info(
                            f"Converted sheet '{sheet_name}' to '{new_csv_path.name}'"
                        )
                except Exception as e:
                    _log.error(f"Excel conversion failed for {file_path.name}: {e}")
                    continue
            else:
                _log.info(f"Skipping unsupported file type: {file_path.name}")

        return processed_csv_files

    input_path = pathlib.Path(input_path)
    input_spreadsheets_files = [input_path / name for name in spreadsheet_split]
    csv_files = convert_excel_to_csv(input_spreadsheets_files, local_processing_dir)
    _log.info(f"CSV files to upload: {[p.name for p in csv_files]}")

    client = LlamaStackClient(base_url=service_url)
    processed = 0

    for csv_path in csv_files:
        content = csv_path.read_text(encoding="utf-8", errors="replace")
        file = client.files.create(
            file=(csv_path.name, io.BytesIO(content.encode("utf-8")), "text/csv"),
            purpose="assistants",
        )
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file.id,
        )
        _log.info(
            f"Uploaded {csv_path.name} (file_id={file.id}) and added to vector store"
        )
        processed += 1

    _log.info(f"Processed {processed} files; added to vector store {vector_store_id}")
    shutil.rmtree(local_processing_dir)


@dsl.pipeline()
def spreadsheet_convert_pipeline(
    base_url: str = "https://raw.githubusercontent.com/opendatahub-io/rag/main/demos/testing-data/spreadsheets",
    spreadsheet_filenames: str = "people.xlsx, sample_sales_data.xlsm, test_customers.csv",
    num_workers: int = 1,
    vector_store_name: str = "csv-vector-store",
    service_url: str = "http://lsd-milvus-service:8321",
    embedding_model_id: str = "sentence-transformers/ibm-granite/granite-embedding-125m-english",
    max_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    use_gpu: bool = False,  # set to True only if you have additional gpu worker
) -> None:
    """
    Converts spreadsheets (Excel to CSV with pandas) and uploads each as a file to a vector store for the Responses API
    :param base_url: Base URL to fetch spreadsheets
    :param spreadsheet_filenames: Comma-separated list of spreadsheets filenames to download and convert
    :param num_workers: Number of parallel workers
    :param vector_store_name: Name of the vector store
    :param service_url: URL of the LlamaStack service
    :param embedding_model_id: Model ID for embedding generation
    :param max_tokens: Maximum number of tokens per chunk
    :param chunk_overlap_tokens: Chunk overlap in tokens
    :param use_gpu: boolean to enable/disable gpu for the convert workers
    :return:
    """
    create_vector_store_task = create_vector_store(
        service_url=service_url,
        vector_store_name=vector_store_name,
        embedding_model_id=embedding_model_id,
        max_tokens=max_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    create_vector_store_task.set_caching_options(False)

    import_task = import_spreadsheet_files(
        base_url=base_url,
        spreadsheet_filenames=spreadsheet_filenames,
    )
    import_task.set_caching_options(True)

    spreadsheet_splits = create_spreadsheet_splits(
        input_path=import_task.output,
        num_splits=num_workers,
    ).set_caching_options(True)

    with dsl.ParallelFor(spreadsheet_splits.output) as spreadsheet_split:
        with dsl.If(use_gpu == True):
            convert_task = convert_and_upload_spreadsheets(
                input_path=import_task.output,
                spreadsheet_split=spreadsheet_split,
                service_url=service_url,
                vector_store_id=create_vector_store_task.output,
            )
            convert_task.set_caching_options(False)
            convert_task.set_cpu_request("500m")
            convert_task.set_cpu_limit("4")
            convert_task.set_memory_request("2Gi")
            convert_task.set_memory_limit("6Gi")
            convert_task.set_accelerator_type("nvidia.com/gpu")
            convert_task.set_accelerator_limit(1)
            add_toleration_json(
                convert_task,
                [
                    {
                        "effect": "NoSchedule",
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                    }
                ],
            )
            add_node_selector_json(convert_task, {})
        with dsl.Else():
            convert_task = convert_and_upload_spreadsheets(
                input_path=import_task.output,
                spreadsheet_split=spreadsheet_split,
                service_url=service_url,
                vector_store_id=create_vector_store_task.output,
            )
            convert_task.set_caching_options(False)
            convert_task.set_cpu_request("500m")
            convert_task.set_cpu_limit("4")
            convert_task.set_memory_request("2Gi")
            convert_task.set_memory_limit("6Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=spreadsheet_convert_pipeline,
        package_path=__file__.replace(".py", "_compiled.yaml"),
    )
