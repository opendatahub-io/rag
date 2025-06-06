# ruff: noqa: PLC0415,UP007,UP035,UP006,E712
# SPDX-License-Identifier: Apache-2.0
from typing import List
import logging

from kfp import compiler, dsl
from kfp.kubernetes import add_node_selector_json, add_toleration_json

PYTHON_BASE_IMAGE = (
    "registry.redhat.io/ubi9/python-312@sha256:e80ff3673c95b91f0dafdbe97afb261eab8244d7fd8b47e20ffcbcfee27fb168"
)

# Workbench Runtime Image: Pytorch with CUDA and Python 3.11 (UBI 9)
# The images for each release can be found in
# https://github.com/red-hat-data-services/rhoai-disconnected-install-helper/blob/main/rhoai-2.21.md
PYTORCH_CUDA_IMAGE = "quay.io/modh/odh-pipeline-runtime-pytorch-cuda-py311-ubi9@sha256:4706be608af3f33c88700ef6ef6a99e716fc95fc7d2e879502e81c0022fd840e"

_log = logging.getLogger(__name__)

@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["gitpython"],
)
def import_test_pdfs(
    input_docs_git_repo: str,
    input_docs_git_branch: str,
    input_docs_git_folder: str,
    output_path: dsl.OutputPath("output-json"),
):
    import os
    import shutil

    from git import Repo

    full_repo_path = os.path.join(output_path, "docling")
    Repo.clone_from(input_docs_git_repo, full_repo_path, branch=input_docs_git_branch)

    # Copy the pdfs the root of the output folder
    pdfs_path = os.path.join(full_repo_path, input_docs_git_folder.lstrip("/"))
    shutil.copytree(pdfs_path, output_path, dirs_exist_ok=True)

    # Delete the repo
    shutil.rmtree(full_repo_path)


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
    splits = [all_pdfs[i::num_splits] for i in range(num_splits)]
    return splits


@dsl.component(
    base_image=PYTORCH_CUDA_IMAGE,
    packages_to_install=["docling", "transformers", "sentence-transformers", "llama-stack", "llama-stack-client", "pymilvus", "fire"],
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
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from docling.chunking import HybridChunker
    import logging
    from llama_stack_client import LlamaStackClient, RAGDocument
    import uuid

    _log = logging.getLogger(__name__)

    # ---- Helper functions ----
    def setup_chunker_and_embedder(embed_model_id: str, max_tokens: int):
        tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
        embedding_model = SentenceTransformer(embed_model_id)
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=True)
        return embedding_model, chunker

    def embed_text(text: str, embedding_model) -> list[float]:
        return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

    def process_and_insert_embeddings(conv_results):
        processed_docs = 0
        for conv_res in conv_results:
            if conv_res.status != ConversionStatus.SUCCESS:
                _log.warning(f"Conversion failed for {conv_res.input.file.stem}: {conv_res.status}")
                continue

            processed_docs += 1
            file_name = conv_res.input.file.stem
            document = conv_res.document
            try:
                document_markdown = document.export_to_markdown()
            except Exception as e:
                _log.warning(f"Failed to export document to markdown: {e}")
                document_markdown = ""

            if document is None:
                _log.warning(f"Document conversion failed for {file_name}")
                continue

            embedding_model, chunker = setup_chunker_and_embedder(embed_model_id, max_tokens)
            for chunk in chunker.chunk(dl_doc=document):
                raw_chunk = chunker.serialize(chunk=chunk)
                embedding = embed_text(raw_chunk, embedding_model)
                
                rag_doc = RAGDocument(
                    document_id=str(uuid.uuid4()),
                    content=raw_chunk,
                    mime_type="text/markdown",
                    metadata={
                        "file_name": file_name,
                        "full_document": document_markdown,
                    },
                    embedding=embedding,
                )

                client.tool_runtime.rag_tool.insert(
                    documents=[rag_doc],
                    vector_db_id=vector_db_id,
                    chunk_size_in_tokens=max_tokens,
                )

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

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
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
    input_docs_git_repo: str = "https://github.com/docling-project/docling",
    input_docs_git_branch: str = "main",
    input_docs_git_folder: str = "/tests/data/pdf/",
    num_workers: int = 1,
    vector_db_id: str = "my_demo_vector_id",
    service_url: str = "http://llama-test-milvus-kserve-service:8321",
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 2048,
    use_gpu: bool = True,
    # tolerations: Optional[list] = [{"effect": "NoSchedule", "key": "nvidia.com/gpu", "operator": "Exists"}],
    # node_selector: Optional[dict] = {},
):
    """
    Converts PDF documents in a git repository to Markdown using Docling and generates embeddings
    :param input_docs_git_repo: git repository containing the documents to convert
    :param input_docs_git_branch: git branch containing the documents to convert
    :param input_docs_git_folder: git folder containing the documents to convert
    :param num_workers: Number of docling worker pods to use
    :param use_gpu: Enable GPU in the docling workers
    :param vector_db_id: ID of the vector database to store embeddings
    :param service_url: URL of the Milvus service
    :return:
    """
    import_task = import_test_pdfs(
        input_docs_git_repo=input_docs_git_repo,
        input_docs_git_branch=input_docs_git_branch,
        input_docs_git_folder=input_docs_git_folder,
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
            convert_task.set_memory_limit("4Gi")
            convert_task.set_accelerator_type("nvidia.com/gpu")
            convert_task.set_accelerator_limit(1)
            add_toleration_json(convert_task, [{"effect": "NoSchedule", "key": "nvidia.com/gpu", "operator": "Exists"}])
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
            convert_task.set_memory_limit("4Gi")

if __name__ == "__main__":
    compiler.Compiler().compile(docling_convert_pipeline, package_path=__file__.replace(".py", "_compiled.yaml"))