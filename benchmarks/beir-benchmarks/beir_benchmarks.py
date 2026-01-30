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

import argparse
import io
import os
import uuid
import pathlib
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

from llama_stack.core.library_client import LlamaStackAsLibraryClient

from llama_stack_client import LlamaStackClient

import numpy as np
import pytrec_eval

import itertools
import logging
import time

DEFAULT_DATASET_NAMES = ["scifact"]
DEFAULT_CUSTOM_DATASETS_URLS = []
DEFAULT_EMBEDDING_MODELS = [
    "sentence-transformers/ibm-granite/granite-embedding-30m-english",
    "sentence-transformers/ibm-granite/granite-embedding-125m-english",
]
DEFAULT_BATCH_SIZE = 150
DEFAULT_VECTOR_DB_PROVIDER_ID = "milvus"
DEFAULT_SEARCH_MODE = "vector"

"""
TODO: Add an arg for specifying the benchmark type when new benchmarks are added.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models with BEIR datasets"
    )

    parser.add_argument(
        "--dataset-names",
        nargs="+",
        type=str,
        default=DEFAULT_DATASET_NAMES,
        help=f"List of BEIR datasets to evaluate (default: {DEFAULT_DATASET_NAMES})",
    )

    parser.add_argument(
        "--custom-datasets-urls",
        nargs="+",
        type=str,
        default=DEFAULT_CUSTOM_DATASETS_URLS,
        help=f"Custom URLs for datasets (default: {DEFAULT_CUSTOM_DATASETS_URLS})",
    )

    parser.add_argument(
        "--embedding-models",
        nargs="+",
        type=str,
        default=DEFAULT_EMBEDDING_MODELS,
        help=f"List of embedding models to evaluate (default: {DEFAULT_EMBEDDING_MODELS})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=150,
        help=f"Batch size for injecting documents (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--vector-db-provider-id",
        type=str,
        default="milvus",
        help=f"Vector DB provider ID (default: {DEFAULT_VECTOR_DB_PROVIDER_ID})",
    )

    parser.add_argument(
        "--search-mode",
        type=str,
        default="vector",
        help=f"Search mode (default: {DEFAULT_SEARCH_MODE})",
    )

    return parser.parse_args()


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

"""
Common classes and functions for BEIR benchmarks.
"""


# LlamaStack RAG Retriever
class LlamaStackRAGRetriever:
    def __init__(
        self,
        llama_stack_client: LlamaStackAsLibraryClient,
        vector_db_id: str,
        top_k: int = 10,
        search_mode: str = "vector",
    ):
        self.llama_stack_client = llama_stack_client
        self.vector_db_id = vector_db_id
        self.top_k = top_k
        self.search_mode = search_mode

    def retrieve(self, queries: dict[str, str], top_k=None):
        results = {}
        times = {}
        top_k = top_k or self.top_k

        for qid, query in queries.items():
            start_time = time.perf_counter()

            rag_results = self.llama_stack_client.vector_stores.search(
                vector_store_id=self.vector_db_id,
                query=query,
                search_mode=self.search_mode,
                max_num_results=top_k,
            )

            end_time = time.perf_counter()
            times[qid] = end_time - start_time

            # Extract corpus document IDs from filenames and use scores directly from search results
            scores = {}
            for result in rag_results.data:
                if result.attributes and "filename" in result.attributes:
                    filename = result.attributes.get("filename")
                    if filename.endswith(".txt"):
                        corpus_doc_id = filename[:-4]  # Remove '.txt'
                        score = (
                            float(result.score)
                            if hasattr(result, "score") and result.score is not None
                            else 0.0
                        )
                        if corpus_doc_id not in scores or score > scores[corpus_doc_id]:
                            scores[corpus_doc_id] = score

            results[qid] = scores

        return results, times


# Load BEIR dataset
def load_beir_dataset(dataset_name: str, custom_datasets_pairs: dict):
    if custom_datasets_pairs == {}:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    else:
        url = custom_datasets_pairs[dataset_name]

    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)

    if not os.path.isdir(data_path):
        data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


# This function is used to inject documents into the LlamaStack vector database.
# The documents are processed in batches to avoid memory issues.
def inject_documents(
    llama_stack_client: LlamaStackAsLibraryClient,
    corpus: dict,
    batch_size: int,
    vector_db_provider_id: str,
    embedding_model: str,
) -> str:
    vector_db_id = f"beir-rag-eval-{embedding_model}-{uuid.uuid4().hex}"

    models = llama_stack_client.models.list()
    em = next(
        (
            m
            for m in models
            if m.custom_metadata.get("model_type") == "embedding"
            and m.id == embedding_model
        ),
        None,
    )
    if em is None:
        raise ValueError(
            f"Embedding model '{embedding_model}' not found or is not an embedding model"
        )
    embedding_model_id = em.id
    embedding_dimension = em.custom_metadata.get("embedding_dimension")

    if embedding_dimension is None:
        raise ValueError(
            f"Embedding dimension not found in model metadata for {embedding_model_id}"
        )

    # Create vector store with empty file_ids initially
    vector_store = llama_stack_client.vector_stores.create(
        name=vector_db_id,
        file_ids=[],
        chunking_strategy={
            "type": "static",
            "static": {
                "max_chunk_size_tokens": 512,
                "chunk_overlap_tokens": 128,
            },
        },
        extra_body={
            "embedding_model": embedding_model_id,
            "embedding_dimension": embedding_dimension,
            "provider_id": vector_db_provider_id,
        },
    )

    # Convert corpus into files and process in batches
    corpus_items = list(corpus.items())
    total_docs = len(corpus_items)

    print(f"Processing {total_docs} documents in batches of {batch_size}")

    for i in range(0, total_docs, batch_size):
        batch_items = corpus_items[i : i + batch_size]
        batch_file_ids = []

        print(
            f"Processing batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(batch_items)} docs)"
        )

        for doc_id, data in batch_items:
            content = data["title"] + " " + data["text"]
            file_content = io.BytesIO(content.encode("utf-8"))
            filename = f"{doc_id}.txt"

            try:
                file = llama_stack_client.files.create(
                    file=(filename, file_content, "text/plain"),
                    purpose="assistants",
                )
                batch_file_ids.append(file.id)
            except Exception as e:
                print(f"ERROR: Failed to create file for document {doc_id}: {str(e)}")
                raise

        # Add files to vector store
        print(
            f"Adding {len(batch_file_ids)} files to vector store (batch {i // batch_size + 1})"
        )
        for file_id in batch_file_ids:
            try:
                llama_stack_client.vector_stores.files.create(
                    vector_store_id=vector_store.id,
                    file_id=file_id,
                )
            except Exception as e:
                print(f"ERROR: Failed to add file {file_id} to vector store: {str(e)}")
                raise
    return vector_store


# Adapted from https://github.com/opendatahub-io/llama-stack-demos/blob/main/demos/rag_eval/Agentic_RAG_with_reference_eval.ipynb
def permutation_test_for_paired_samples(
    scores_a: list[float], scores_b: list[float], iterations: int = 10_000
):
    """
    Performs a permutation test of a given statistic on provided data.
    """

    from scipy.stats import permutation_test

    def _statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    result = permutation_test(
        data=(scores_a, scores_b),
        statistic=_statistic,
        n_resamples=iterations,
        alternative="two-sided",
        permutation_type="samples",
    )
    return float(result.pvalue)


# Adapted from https://github.com/opendatahub-io/llama-stack-demos/blob/main/demos/rag_eval/Agentic_RAG_with_reference_eval.ipynb
def print_stats_significance(
    scores_a: list, scores_b: list, overview_label: str, label_a: str, label_b: str
):
    mean_score_a = np.mean(scores_a)
    mean_score_b = np.mean(scores_b)

    p_value = permutation_test_for_paired_samples(scores_a, scores_b)
    print(overview_label)
    print(f" {label_a:<50}: {mean_score_a:>10.4f}")
    print(f" {label_b:<50}: {mean_score_b:>10.4f}")
    print(f" {'p_value':<50}: {p_value:>10.4f}")

    if p_value < 0.05:
        print("  p_value<0.05 so this result is statistically significant")
        # Note that the logic below is incorrect if the mean scores are equal, but that can't be true if p<1.
        higher_model_id = label_a if mean_score_a >= mean_score_b else label_b
        print(
            f"  You can conclude that {higher_model_id} retrieval is better on data of this sort"
        )
    else:
        import math

        print("  p_value>=0.05 so this result is NOT statistically significant.")
        print(
            "  You can conclude that there is not enough data to tell which is better."
        )
        num_samples = len(scores_a)
        margin_of_error = 1 / math.sqrt(num_samples)
        print(
            f"  Note that this data includes {num_samples} questions which typically produces a margin of error of around +/-{margin_of_error:.1%}."
        )
        print("  So the two are probably roughly within that margin of error or so.")


def get_metrics(all_scores: dict):
    for scores_for_dataset in all_scores.values():
        for scores_for_condition in scores_for_dataset.values():
            for scores_for_question in scores_for_condition.values():
                metrics = list(scores_for_question.keys())
                metrics.sort()
                return metrics
    return []


def print_scores(all_scores: dict):
    metrics = get_metrics(all_scores)
    for dataset_name, scores_for_dataset in all_scores.items():
        condition_labels = list(scores_for_dataset.keys())
        condition_labels.sort()
        for metric in metrics:
            overview_label = f"{dataset_name} {metric}"
            for label_a, label_b in itertools.combinations(condition_labels, 2):
                scores_for_label_a = scores_for_dataset[label_a]
                scores_for_label_b = scores_for_dataset[label_b]
                scores_a = [
                    score_group[metric] for score_group in scores_for_label_a.values()
                ]
                scores_b = [
                    score_group[metric] for score_group in scores_for_label_b.values()
                ]
                print_stats_significance(
                    scores_a, scores_b, overview_label, label_a, label_b
                )
                print("\n")


"""
The BenchmarkEmbeddingModels class is used to evaluate the retrieval performance of the embedding models using the generic LlamaStack RAG Retriever.
"""


class BenchmarkEmbeddingModels:
    def __init__(
        self,
        llama_stack_client: LlamaStackAsLibraryClient,
        datasets: list[str],
        custom_datasets_urls: list[str],
        batch_size: int,
        vector_db_provider_id: str,
        embedding_models: list[str],
        search_mode: str,
    ):
        self.llama_stack_client = llama_stack_client
        self.datasets = datasets
        self.custom_datasets_urls = custom_datasets_urls
        self.batch_size = batch_size
        self.vector_db_provider_id = vector_db_provider_id
        self.embedding_models = embedding_models
        self.search_mode = search_mode

    def evaluate_retrieval(
        self,
    ):
        results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
        # Create results directory at the start to ensure it exists
        os.makedirs(results_dir, exist_ok=True)
        all_scores = {}

        custom_datasets_pairs = {}
        if self.custom_datasets_urls:
            custom_datasets_pairs = {
                dataset_name: self.custom_datasets_urls[i]
                for i, dataset_name in enumerate(self.datasets)
            }

        for dataset_name in self.datasets:
            all_scores[dataset_name] = {}
            corpus, queries, qrels = load_beir_dataset(
                dataset_name, custom_datasets_pairs
            )
            for embedding_model in self.embedding_models:
                print(
                    f"\n====================== {dataset_name}, {embedding_model} ======================"
                )
                print(f"Ingesting {dataset_name}, {embedding_model}")
                vector_store = inject_documents(
                    self.llama_stack_client,
                    corpus,
                    self.batch_size,
                    self.vector_db_provider_id,
                    embedding_model,
                )

                retriever = LlamaStackRAGRetriever(
                    self.llama_stack_client,
                    vector_store.id,
                    top_k=10,
                    search_mode=self.search_mode,
                )

                print("Retrieving")
                results, times = retriever.retrieve(queries, top_k=10)

                print("Scoring")
                k_values = [5, 10]

                # This is a subset of the evaluation metrics used in beir.retrieval.evaluation.
                # It formulates the metric strings at https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py#L61
                # and then calls pytrec_eval.RelevanceEvaluator.  We call pytrec_eval.RelevanceEvaluator directly using some of
                # those strings because we want not only the overall averages (which beir.retrieval.evaluation provides) but also
                # the scores for each question so we can compute statistical significance.
                map_string = "map_cut." + ",".join([str(k) for k in k_values])
                ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
                metrics_strings = {ndcg_string, map_string}

                evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_strings)
                scores = evaluator.evaluate(results)
                for qid, scores_for_qid in scores.items():
                    scores_for_qid["time"] = times[qid]

                all_scores[dataset_name][embedding_model] = scores

                safe_embedding_model = embedding_model.replace("/", "_")
                util.save_runfile(
                    os.path.join(
                        results_dir,
                        f"{dataset_name}-{vector_store.id}-{safe_embedding_model}-{self.search_mode}.run.trec",
                    ),
                    results,
                )
        print(f"All results in {results_dir}\n")

        return all_scores


if __name__ == "__main__":
    args = parse_args()

    # A check for when custom dataset urls are set they are compared with the number of dataset names
    if args.custom_datasets_urls and len(args.custom_datasets_urls) != len(
        args.dataset_names
    ):
        raise ValueError(
            f"Number of custom dataset URLs ({len(args.custom_datasets_urls)}) must match "
            f"number of dataset names ({len(args.dataset_names)}). "
            f"Got URLs: {args.custom_datasets_urls}, dataset names: {args.dataset_names}"
        )

    # Run LlamaStack Client
    llama_stack_url = os.environ.get("LLAMA_STACK_URL")
    if llama_stack_url:
        llama_stack_client = LlamaStackClient(base_url=llama_stack_url)
    else:
        print(
            "No Llama Stack URL provided, please set the LLAMA_STACK_URL environment variable"
        )
        exit(1)

    """
    TODO: When adding a new benchmark add a check to see which of the available benchmarks to use and set a generic variable to the benchmark class.
    e.g.
    benchmark = None
    if args.benchmark_type == "embedding_models":
        benchmark = BenchmarkEmbeddingModels(
            llama_stack_client,
            args.dataset_names,
            args.custom_datasets_urls,
            args.batch_size,
            "milvus",
            args.embedding_models,
        )
    elif args.benchmark_type == "benchmark2":
        benchmark = Benchmark2(...)

    benchmark.evaluate_retrieval(...)    
    """
    embedding_models_benchmark = BenchmarkEmbeddingModels(
        llama_stack_client,
        args.dataset_names,
        args.custom_datasets_urls,
        args.batch_size,
        args.vector_db_provider_id,
        args.embedding_models,
        args.search_mode,
    )
    all_scores = embedding_models_benchmark.evaluate_retrieval()
    print_scores(all_scores)

"""
Scoring for embedding models benchmark:
All results in <path-to>/rag/benchmarks/embedding-models-with-beir/results

scifact map_cut_10
 granite-embedding-125m                            :     0.6879
 granite-embedding-30m                             :     0.6578
 p_value                                           :     0.0150
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


scifact map_cut_5
 granite-embedding-125m                            :     0.6767
 granite-embedding-30m                             :     0.6481
 p_value                                           :     0.0294
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


scifact ndcg_cut_10
 granite-embedding-125m                            :     0.7350
 granite-embedding-30m                             :     0.7018
 p_value                                           :     0.0026
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


scifact ndcg_cut_5
 granite-embedding-125m                            :     0.7119
 granite-embedding-30m                             :     0.6833
 p_value                                           :     0.0256
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


fiqa map_cut_10
 granite-embedding-125m                            :     0.3581
 granite-embedding-30m                             :     0.2829
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


fiqa map_cut_5
 granite-embedding-125m                            :     0.3395
 granite-embedding-30m                             :     0.2664
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


fiqa ndcg_cut_10
 granite-embedding-125m                            :     0.4411
 granite-embedding-30m                             :     0.3599
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


fiqa ndcg_cut_5
 granite-embedding-125m                            :     0.4176
 granite-embedding-30m                             :     0.3355
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


arguana map_cut_10
 granite-embedding-125m                            :     0.2927
 granite-embedding-30m                             :     0.2821
 p_value                                           :     0.0104
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


arguana map_cut_5
 granite-embedding-125m                            :     0.2707
 granite-embedding-30m                             :     0.2594
 p_value                                           :     0.0216
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


arguana ndcg_cut_10
 granite-embedding-125m                            :     0.4251
 granite-embedding-30m                             :     0.4124
 p_value                                           :     0.0044
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort


arguana ndcg_cut_5
 granite-embedding-125m                            :     0.3718
 granite-embedding-30m                             :     0.3582
 p_value                                           :     0.0292
  p_value<0.05 so this result is statistically significant
  You can conclude that granite-embedding-125m retrieval is better on data of this sort
"""
