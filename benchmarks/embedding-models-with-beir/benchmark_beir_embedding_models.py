import argparse
import os
import uuid
import pathlib
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack.apis.tools import RAGQueryConfig
from llama_stack_client.types import Document

import numpy as np
import pytrec_eval

import itertools
import logging

DEFAULT_DATASET_NAMES = ["scifact"]
DEFAULT_CUSTOM_DATASETS_URLS = []
DEFAULT_EMBEDDING_MODELS = ["granite-embedding-30m", "granite-embedding-125m"]
DEFAULT_BATCH_SIZE = 150


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models with BEIR datasets"
    )

    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=DEFAULT_DATASET_NAMES,
        help=f"List of BEIR datasets to evaluate (default: {DEFAULT_DATASET_NAMES})",
    )

    parser.add_argument(
        "--custom-datasets-urls",
        nargs="+",
        default=DEFAULT_CUSTOM_DATASETS_URLS,
        help=f"Custom URLs for datasets (default: {DEFAULT_CUSTOM_DATASETS_URLS})",
    )

    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=DEFAULT_EMBEDDING_MODELS,
        help=f"List of embedding models to evaluate (default: {DEFAULT_EMBEDDING_MODELS})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=150,
        help=f"Batch size for injecting documents (default: {DEFAULT_BATCH_SIZE})",
    )

    return parser.parse_args()


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


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


"""
This function is used to inject documents into the LlamaStack vector database.
The documents are processed in batches to avoid memory issues.
"""


def inject_documents(
    llama_stack_client, corpus, batch_size, vector_db_provider_id, embedding_model
):
    vector_db_id = f"beir-rag-eval-{embedding_model}-{uuid.uuid4().hex}"

    llama_stack_client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model,
        provider_id=vector_db_provider_id,
    )

    # Convert corpus into Documents and process in batches
    corpus_items = list(corpus.items())
    total_docs = len(corpus_items)

    print(f"Processing {total_docs} documents in batches of {batch_size}")

    for i in range(0, total_docs, batch_size):
        batch_items = corpus_items[i : i + batch_size]
        documents_batch = [
            Document(
                document_id=doc_id,
                content=data["title"] + " " + data["text"],
                mime_type="text/plain",
                metadata={},
            )
            for doc_id, data in batch_items
        ]

        print(
            f"Inserting batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(documents_batch)} docs)"
        )

        llama_stack_client.tool_runtime.rag_tool.insert(
            documents=documents_batch,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=512,
            timeout=3600,
        )

    print(f"Successfully inserted all {total_docs} documents")
    return vector_db_id


# LlamaStack RAG Retriever
class LlamaStackRAGRetriever:
    def __init__(self, vector_db_id, query_config, top_k=10):
        self.llama_stack_client = llama_stack_client
        self.vector_db_id = vector_db_id
        self.query_config = query_config
        self.top_k = top_k

    def retrieve(self, queries, top_k=None):
        results = {}
        top_k = top_k or self.top_k

        for qid, query in queries.items():
            rag_results = self.llama_stack_client.tool_runtime.rag_tool.query(
                vector_db_ids=[self.vector_db_id],
                content=query,
                query_config={**self.query_config, "max_chunks": top_k},
            )

            doc_ids = rag_results.metadata.get("document_ids", [])
            scores = {doc_id: 1.0 - (i * 0.01) for i, doc_id in enumerate(doc_ids)}

            results[qid] = scores

        return results


# Adapted from https://github.com/opendatahub-io/llama-stack-demos/blob/main/demos/rag_eval/Agentic_RAG_with_reference_eval.ipynb
def permutation_test_for_paired_samples(scores_a, scores_b, iterations=10_000):
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
def print_stats_significance(scores_a, scores_b, overview_label, label_a, label_b):
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
            f"  You can conclude that {higher_model_id} generation is better on data of this sort"
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


def get_metrics(all_scores):
    for scores_for_dataset in all_scores.values():
        for scores_for_condition in scores_for_dataset.values():
            for scores_for_question in scores_for_condition.values():
                metrics = list(scores_for_question.keys())
                metrics.sort()
                return metrics
    return []


def print_scores(all_scores):
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


def evaluate_retrieval(
    llama_stack_client,
    datasets,
    custom_datasets_urls,
    batch_size,
    vector_db_provider_id,
    embedding_models,
):
    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    all_scores = {}

    custom_datasets_pairs = {}
    if custom_datasets_urls:
        custom_datasets_pairs = {
            dataset_name: custom_datasets_urls[i]
            for i, dataset_name in enumerate(datasets)
        }

    for dataset_name in datasets:
        all_scores[dataset_name] = {}
        corpus, queries, qrels = load_beir_dataset(dataset_name, custom_datasets_pairs)
        for embedding_model in embedding_models:
            print(
                f"\n====================== {dataset_name}, {embedding_model} ======================"
            )
            print(f"Ingesting {dataset_name}, {embedding_model}")
            vector_db_id = inject_documents(
                llama_stack_client,
                corpus,
                batch_size,
                vector_db_provider_id,
                embedding_model,
            )

            query_config = RAGQueryConfig(max_chunks=10, mode="vector").model_dump()
            retriever = LlamaStackRAGRetriever(vector_db_id, query_config, top_k=10)

            print("Retrieving")
            results = retriever.retrieve(queries, top_k=10)

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

            all_scores[dataset_name][embedding_model] = scores

            os.makedirs(results_dir, exist_ok=True)
            util.save_runfile(
                os.path.join(
                    results_dir,
                    f"{dataset_name}-{vector_db_id}-{embedding_model}.run.trec",
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

    llama_stack_client = LlamaStackAsLibraryClient("./run.yaml")
    llama_stack_client.initialize()
    all_scores = evaluate_retrieval(
        llama_stack_client,
        args.dataset_names,
        args.custom_datasets_urls,
        args.batch_size,
        "milvus",
        args.embedding_models,
    )
    print_scores(all_scores)
