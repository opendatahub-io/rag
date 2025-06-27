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

# Workbench Runtime Image: Pytorch with CUDA and Python 3.11 (UBI 9)
# The images for each release can be found in
# https://github.com/red-hat-data-services/rhoai-disconnected-install-helper/blob/main/rhoai-2.21.md
PYTORCH_CUDA_IMAGE = "quay.io/modh/odh-pipeline-runtime-pytorch-cuda-py311-ubi9@sha256:4706be608af3f33c88700ef6ef6a99e716fc95fc7d2e879502e81c0022fd840e"

_log = logging.getLogger(__name__)


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["llama-stack-client", "fire", "requests"],
)
def register_vector_db(
    service_url: str,
    vector_db_id: str,
    embed_model_id: str,
):
    from llama_stack_client import LlamaStackClient

    client = LlamaStackClient(base_url=service_url)

    models = client.models.list()
    matching_model = next(
        (m for m in models if m.provider_resource_id == embed_model_id), None
    )

    if not matching_model:
        raise ValueError(
            f"Model with ID '{embed_model_id}' not found on LlamaStack server."
        )

    if matching_model.model_type != "embedding":
        raise ValueError(f"Model '{embed_model_id}' is not an embedding model")

    embedding_dimension = matching_model.metadata["embedding_dimension"]

    # Register the vector DB
    _ = client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=matching_model.identifier,
        embedding_dimension=embedding_dimension,
        provider_id="milvus",
    )
    print(
        f"Registered vector DB '{vector_db_id}' with embedding model '{embed_model_id}'."
    )


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["requests"],
)
def import_test_pdfs(
    base_url: str,
    pdf_filenames: str,
    output_path: dsl.OutputPath("input-pdfs"),
):
    import os
    import requests
    import shutil

    os.makedirs(output_path, exist_ok=True)
    filenames = [f.strip() for f in pdf_filenames.split(",") if f.strip()]

    for filename in filenames:
        url = f"{base_url.rstrip('/')}/{filename}"
        file_path = os.path.join(output_path, filename)

        try:
            with requests.get(url, stream=True, timeout=10) as response:
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    shutil.copyfileobj(response.raw, f)
            print(f"Downloaded {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}: {e}, skipping.")


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
)
def create_pdf_splits(
    input_path: dsl.InputPath("input-pdfs"),
    num_splits: int,
) -> List[List[str]]:
    import pathlib

    # Split our entire directory of pdfs into n batches, where n == num_splits
    all_pdfs = [path.name for path in pathlib.Path(input_path).glob("*.pdf")]
    splits = [
        batch for batch in (all_pdfs[i::num_splits] for i in range(num_splits)) if batch
    ]
    return splits or [[]]


# This component converts PDFs to Markdown and ingests the embeddings into LlamaStack's vector store
@dsl.component(
    base_image=PYTORCH_CUDA_IMAGE,
    packages_to_install=[
        "docling>=2.43.0",
        "transformers",
        "sentence-transformers",
        "llama-stack",
        "llama-stack-client",
        "pymilvus",
        "fire",
        "rapidocr-onnxruntime",
    ],
)
def docling_convert(
    input_path: dsl.InputPath("input-pdfs"),
    pdf_split: List[str],
    output_path: dsl.OutputPath("output-md"),
    embed_model_id: str,
    max_tokens: int,
    service_url: str,
    vector_db_id: str,
):
    import pathlib

    from docling.datamodel.base_models import InputFormat, ConversionStatus
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from docling.chunking import HybridChunker
    import logging
    from llama_stack_client import LlamaStackClient
    import uuid

    import json

    _log = logging.getLogger(__name__)

    # ---- Helper functions ----
    def setup_chunker_and_embedder(embed_model_id: str, max_tokens: int):
        tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
        embedding_model = SentenceTransformer(embed_model_id)
        chunker = HybridChunker(
            tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=True
        )
        return embedding_model, chunker

    def embed_text(text: str, embedding_model) -> list[float]:
        return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

    def process_and_insert_embeddings(conv_results):
        processed_docs = 0
        for conv_res in conv_results:
            if conv_res.status != ConversionStatus.SUCCESS:
                _log.warning(
                    f"Conversion failed for {conv_res.input.file.stem}: {conv_res.status}"
                )
                continue

            processed_docs += 1
            file_name = conv_res.input.file.stem
            document = conv_res.document

            if document is None:
                _log.warning(f"Document conversion failed for {file_name}")
                continue

            embedding_model, chunker = setup_chunker_and_embedder(
                embed_model_id, max_tokens
            )

            chunks_with_embedding = []
            for chunk in chunker.chunk(dl_doc=document):
                raw_chunk = chunker.contextualize(chunk)
                embedding = embed_text(raw_chunk, embedding_model)

                chunk_id = str(uuid.uuid4())  # Generate a unique ID for the chunk
                content_token_count = chunker.tokenizer.count_tokens(raw_chunk)

                # Prepare metadata object
                metadata_obj = {
                    "file_name": file_name,
                    "document_id": chunk_id,
                    "token_count": content_token_count,
                }

                metadata_str = json.dumps(metadata_obj)
                metadata_token_count = chunker.tokenizer.count_tokens(metadata_str)
                metadata_obj["metadata_token_count"] = metadata_token_count

                chunks_with_embedding.append(
                    {
                        "content": raw_chunk,
                        "mime_type": "text/markdown",
                        "embedding": embedding,
                        "metadata": metadata_obj,
                    }
                )
            if chunks_with_embedding:
                try:
                    client.vector_io.insert(
                        vector_db_id=vector_db_id, chunks=chunks_with_embedding
                    )
                except Exception as e:
                    _log.error(f"Failed to insert embeddings into vector database: {e}")

        _log.info(f"Processed {processed_docs} documents successfully.")

    # ---- Main logic ----
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Original code using splits
    input_pdfs = [input_path / name for name in pdf_split]
    # Alternative not using splits
    # input_pdfs = pathlib.Path(input_path).glob("*.pdf")

    # Required models are automatically downloaded when they are
    # not provided in PdfPipelineOptions initialization
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.generate_page_images = True
    pipeline_options.ocr_options = RapidOcrOptions()

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_results = doc_converter.convert_all(
        input_pdfs,
        raises_on_error=True,
    )

    # Initialize LlamaStack client
    client = LlamaStackClient(base_url=service_url)

    # Process the conversion results and insert embeddings into the vector database
    process_and_insert_embeddings(conv_results)


@dsl.pipeline()
def docling_convert_pipeline(
    base_url: str = "https://raw.githubusercontent.com/docling-project/docling/main/tests/data/pdf",
    pdf_filenames: str = "2203.01017v2.pdf, 2206.01062.pdf, 2305.03393v1-pg9.pdf, amt_handbook_sample.pdf, code_and_formula.pdf, multi_page.pdf, picture_classification.pdf, redp5110_sampled.pdf, right_to_left_01.pdf, right_to_left_02.pdf, right_to_left_03.pdf",
    num_workers: int = 1,
    vector_db_id: str = "my_demo_vector_id",
    service_url: str = "http://lsd-llama-milvus-service:8321",
    embed_model_id: str = "ibm-granite/granite-embedding-125m-english",
    max_tokens: int = 512,
    use_gpu: bool = True,
    # tolerations: Optional[list] = [{"effect": "NoSchedule", "key": "nvidia.com/gpu", "operator": "Exists"}],
    # node_selector: Optional[dict] = {},
):
    """
    Converts PDF documents in a git repository to Markdown using Docling and generates embeddings
    :param base_url: Base URL to fetch PDF files from
    :param pdf_filenames: Comma-separated list of PDF filenames to download and convert
    :param num_workers: Number of docling worker pods to use
    :param use_gpu: Enable GPU in the docling workers
    :param vector_db_id: ID of the vector database to store embeddings
    :param service_url: URL of the Milvus service
    :param embed_model_id: Model ID for embedding generation
    :param max_tokens: Maximum number of tokens per chunk
    :return:
    """

    register_task = register_vector_db(
        service_url=service_url,
        vector_db_id=vector_db_id,
        embed_model_id=embed_model_id,
    )
    register_task.set_caching_options(False)

    import_task = import_test_pdfs(
        base_url=base_url,
        pdf_filenames=pdf_filenames,
    )
    import_task.set_caching_options(True)

    pdf_splits = create_pdf_splits(
        input_path=import_task.output,
        num_splits=num_workers,
    ).set_caching_options(True)

    with dsl.ParallelFor(pdf_splits.output) as pdf_split:
        with dsl.If(use_gpu == True):
            convert_task = docling_convert(
                input_path=import_task.output,
                pdf_split=pdf_split,
                embed_model_id=embed_model_id,
                max_tokens=max_tokens,
                service_url=service_url,
                vector_db_id=vector_db_id,
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
            convert_task = docling_convert(
                input_path=import_task.output,
                pdf_split=pdf_split,
                embed_model_id=embed_model_id,
                max_tokens=max_tokens,
                service_url=service_url,
                vector_db_id=vector_db_id,
            )
            convert_task.set_caching_options(False)
            convert_task.set_cpu_request("500m")
            convert_task.set_cpu_limit("4")
            convert_task.set_memory_request("2Gi")
            convert_task.set_memory_limit("6Gi")


if __name__ == "__main__":
    compiler.Compiler().compile(
        docling_convert_pipeline, package_path=__file__.replace(".py", "_compiled.yaml")
    )
