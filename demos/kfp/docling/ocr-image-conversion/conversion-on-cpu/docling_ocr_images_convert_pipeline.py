# ruff: noqa: PLC0415,UP007,UP035,UP006,E712
# SPDX-License-Identifier: Apache-2.0
from typing import List, Iterator, Tuple
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
) -> None:
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
def import_test_images(
    base_url: str,
    image_filenames: str,
    output_path: dsl.OutputPath("input-images"),
) -> None:
    import os
    import requests
    import shutil

    os.makedirs(output_path, exist_ok=True)
    filenames = [f.strip() for f in image_filenames.split(",") if f.strip()]

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
def create_image_splits(
    input_path: dsl.InputPath("input-images"),
    num_splits: int,
) -> List[List[str]]:
    import pathlib

    # Split our entire directory of images into n batches, where n == num_splits
    # Support multiple image formats
    all_images = []
    image_extensions = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.tiff",
        "*.bmp",
        "*.webp",
    ]

    for ext in image_extensions:
        all_images.extend([path.name for path in pathlib.Path(input_path).glob(ext)])

    splits = [
        batch
        for batch in (all_images[i::num_splits] for i in range(num_splits))
        if batch
    ]
    return splits or [[]]


@dsl.component(
    base_image=PYTORCH_CUDA_IMAGE,
    packages_to_install=[
        "docling",
        "transformers",
        "sentence-transformers",
        "llama-stack",
        "llama-stack-client",
        "pymilvus",
        "fire",
        "requests",
        "Pillow",
        "huggingface_hub",
    ],
)
def docling_convert_and_ingest_images(
    input_path: dsl.InputPath("input-images"),
    image_split: List[str],
    output_path: dsl.OutputPath("output-md"),
    embed_model_id: str,
    max_tokens: int,
    service_url: str,
    vector_db_id: str,
) -> None:
    import pathlib
    import json
    import uuid
    import logging
    import os
    from docling.datamodel.pipeline_options import RapidOcrOptions
    from huggingface_hub import snapshot_download
    from docling.datamodel.base_models import InputFormat, ConversionStatus
    from docling.datamodel.document import ConversionResult
    from docling.document_converter import DocumentConverter, ImageFormatOption
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from llama_stack_client import LlamaStackClient

    _log = logging.getLogger(__name__)

    def setup_chunker_and_embedder(
        embed_model_id: str, max_tokens: int
    ) -> Tuple[SentenceTransformer, HybridChunker]:
        """Setup embedding model and chunker for text processing"""
        tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
        embedding_model = SentenceTransformer(embed_model_id)
        chunker = HybridChunker(
            tokenizer=tokenizer, max_tokens=max_tokens, merge_peers=True
        )
        return embedding_model, chunker

    def embed_text(text: str, embedding_model: SentenceTransformer) -> List[float]:
        """Generate embeddings for text content"""
        return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

    def process_and_insert_embeddings(
        conv_results: Iterator[ConversionResult], client: LlamaStackClient
    ) -> None:
        processed_docs = 0
        embedding_model, chunker = setup_chunker_and_embedder(
            embed_model_id, max_tokens
        )

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

            chunks_with_embedding = []
            for chunk in chunker.chunk(dl_doc=document):
                raw_chunk = chunker.contextualize(chunk)
                embedding = embed_text(raw_chunk, embedding_model)
                chunk_id = str(uuid.uuid4())
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

                # Create a new embedding object with the required fields
                new_embedding = {
                    "content": raw_chunk,
                    "mime_type": "text/markdown",
                    "embedding": embedding,
                    "metadata": metadata_obj,
                }

                print(f"New embedding: {new_embedding}")

                chunks_with_embedding.append(new_embedding)

            if chunks_with_embedding:
                try:
                    client.vector_io.insert(
                        vector_db_id=vector_db_id, chunks=chunks_with_embedding
                    )
                except Exception as e:
                    _log.error(f"Failed to insert embeddings into vector database: {e}")

        _log.info(f"Processed {processed_docs} documents successfully.")

    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    input_images = [input_path / name for name in image_split]

    # Download RapidOCR models from HuggingFace
    print("Downloading RapidOCR models")
    download_path = snapshot_download(repo_id="SWHL/RapidOCR")

    # Setup RapidOcrOptions for english detection
    det_model_path = os.path.join(
        download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
    )

    image_format_options = ImageFormatOption(
        do_ocr=True,
        ocr_options=RapidOcrOptions(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            cls_model_path=cls_model_path,
        ),
    )

    # Create Docling OCR converter
    docling_ocr_converter = DocumentConverter(
        format_options={InputFormat.IMAGE: image_format_options}
    )

    # Convert all image files to text
    conv_results = docling_ocr_converter.convert_all(
        input_images,
        raises_on_error=True,
    )

    client = LlamaStackClient(base_url=service_url)

    process_and_insert_embeddings(conv_results, client)


@dsl.pipeline()
def docling_convert_pipeline(
    base_url: str = "https://raw.githubusercontent.com/opendatahub-io/rag/main/demos/testing-data/images",
    image_filenames: str = "RAG_flow_diagram.png, RAG_key_market_usecases.png",
    num_workers: int = 1,
    vector_db_id: str = "ocr-vector-db",
    service_url: str = "http://lsd-llama-milvus-service:8321",
    embed_model_id: str = "ibm-granite/granite-embedding-125m-english",
    max_tokens: int = 512,
    use_gpu: bool = False,  # use only if you have additional gpu worker
) -> None:
    """
    Converts image documents to text using Docling OCR and generates embeddings
    :param base_url: Base URL to fetch image files from
    :param image_filenames: Comma-separated list of image filenames to download and convert
    :param num_workers: Number of docling worker pods to use
    :param vector_db_id: ID of the vector database to store embeddings
    :param service_url: URL of the LlamaStack service
    :param embed_model_id: Model ID for embedding generation
    :param max_tokens: Maximum number of tokens per chunk
    :param use_gpu: boolean to enable/disable gpu in the docling workers
    :return:
    """

    register_task = register_vector_db(
        service_url=service_url,
        vector_db_id=vector_db_id,
        embed_model_id=embed_model_id,
    )
    register_task.set_caching_options(False)

    import_task = import_test_images(
        base_url=base_url,
        image_filenames=image_filenames,
    )
    import_task.set_caching_options(True)

    image_splits = create_image_splits(
        input_path=import_task.output,
        num_splits=num_workers,
    ).set_caching_options(True)

    with dsl.ParallelFor(image_splits.output) as image_split:
        with dsl.If(use_gpu == True):
            convert_task = docling_convert_and_ingest_images(
                input_path=import_task.output,
                image_split=image_split,
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
            convert_task = docling_convert_and_ingest_images(
                input_path=import_task.output,
                image_split=image_split,
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
        pipeline_func=docling_convert_pipeline,
        package_path=__file__.replace(".py", "_compiled.yaml"),
    )
