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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import itertools
import logging
import os
import pathlib
import sys
import time
import uuid

import numpy as np
import pytrec_eval
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from llama_stack_client.types import Document
from pymilvus import MilvusClient, model

from llama_stack.apis.tools import RAGQueryConfig
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack.models.llama.llama3.tokenizer import Tokenizer

"""
Test script to benchmark retrievel accuracy with and without Llama Stack using the BEIR IR datasets.
If everything is working as intended, it will show no difference with and without Llama Stack.
In contrast, if there is some sort of defect in either Llama Stack or the alternative implementation.

This test script uses the following configuration:
- Milvus inline vector DB provider
- SentenceTransformer embedding inference provider
- ibm-granite/granite-embedding-125m-english embedding model.  See https://github.com/instructlab/dev-docs/blob/main/docs/rag/adrs/granite-embeddings.md for an argument why this is a good default model.
- 10 chunks retrieved
- Max chunk size of 512 tokens

In the Llama Stack test case, all three of these are called through the Llama Stack APIs.  In the alternative,
all of these are called through the PyMilvus API.

At the time that this script is being written, it shows no difference between the Llama Stack test case
(which it labels LlamaStackRAGRetriever) and the alternative (which it labels MilvusRetriever).  If it
does show a significant difference in the future then one or both of those might not be working as
intended any more.

The script succeeds with a return code of 0 if no significant difference is detected and fails with a
return code of 1 if any significant difference is detected.

The run.yaml file in this directory includes the providers and model needed to run these tests.
Before running this script, start a Llama Stack server on your local machine with the default
port and that run.yaml.
"""


# We are running with scifact, which has the smallest number of documents of the BEIR data sets (only 5K documents).
# Most of the time is spent vectorizing and ingesting the documents, so running on the smallest number of documents
# makes this assessment faster.  Running on more datasets would make it more robust.  So it is a tricky trade-off.
# See a full list of available datasets at https://github.com/beir-cellar/beir?tab=readme-ov-file#beers-available-datasets
DATASETS = ["scifact"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure the internal logging from the beir library
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


# Load BEIR dataset
def load_beir_dataset(dataset_name: str):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")

    data_path = os.path.join(out_dir, dataset_name)
    print(data_path)
    if not os.path.isdir(data_path):
        data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


# Inject documents into LlamaStack vector database
def inject_documents_llama_stack(
    llama_stack_client,
    corpus,
    vector_db_provider_id,
    embedding_model_id,
    chunk_size_in_tokens,
):
    vector_db_id = f"beir-rag-eval-{uuid.uuid4().hex}"

    embedding_dimension = None
    for m in llama_stack_client.models.list():
        if (
            m.api_model_type == "embedding"
            and m.provider_resource_id == embedding_model_id
        ):
            embedding_dimension = m.metadata["embedding_dimension"]
            embedding_model_name = m.identifier

    llama_stack_client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_name,
        provider_id=vector_db_provider_id,
        embedding_dimension=embedding_dimension,
    )

    # Convert corpus into Documents
    documents = [
        Document(
            document_id=doc_id,
            content=data["title"] + " " + data["text"],
            mime_type="text/plain",
            metadata={},
        )
        for doc_id, data in corpus.items()
    ]

    llama_stack_client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=chunk_size_in_tokens,
        timeout=36000,
    )

    return vector_db_id


def llama_stack_style_chunker(text, chunk_size_in_tokens):
    """
    Llama Stack style chunker.
    To provide a fair comparison with Llama Stack we try to replicate the behavior of the Llama Stack chunker.  Here are the relevant code snippets:

    https://github.com/meta-llama/llama-stack/blob/2603f10f95fcd302297158adb709d2a84c9f60af/llama_stack/providers/inline/tool_runtime/rag/memory.py#L84C1-L92C14
    chunks.extend(
        make_overlapped_chunks(
            doc.document_id,
            content,
            chunk_size_in_tokens,
            chunk_size_in_tokens // 4,
            doc.metadata,
        )
    )

    https://github.com/meta-llama/llama-stack/blob/2603f10f95fcd302297158adb709d2a84c9f60af/llama_stack/providers/utils/memory/vector_store.py#L142
    def make_overlapped_chunks(
        document_id: str, text: str, window_len: int, overlap_len: int, metadata: dict[str, Any]
    ) -> list[Chunk]:
        tokenizer = Tokenizer.get_instance()
        tokens = tokenizer.encode(text, bos=False, eos=False)
        try:
            metadata_string = str(metadata)
        except Exception as e:
            raise ValueError("Failed to serialize metadata to string") from e

        metadata_tokens = tokenizer.encode(metadata_string, bos=False, eos=False)

        chunks = []
        for i in range(0, len(tokens), window_len - overlap_len):
            toks = tokens[i : i + window_len]
            chunk = tokenizer.decode(toks)
            chunk_metadata = metadata.copy()
            chunk_metadata["document_id"] = document_id
            chunk_metadata["token_count"] = len(toks)
            chunk_metadata["metadata_token_count"] = len(metadata_tokens)

            # chunk is a string
            chunks.append(
                Chunk(
                    content=chunk,
                    metadata=chunk_metadata,
                )
            )

        return chunks

    This does the same thing except without the metadata and Chunk object wrapper which are not needed for these tests.
    """
    window_len = chunk_size_in_tokens
    overlap_len = chunk_size_in_tokens // 4

    tokenizer = Tokenizer.get_instance()
    tokens = tokenizer.encode(text, bos=False, eos=False)
    chunks = []
    for i in range(0, len(tokens), window_len - overlap_len):
        toks = tokens[i : i + window_len]
        chunk = tokenizer.decode(toks)
        chunks.append(chunk)
    return chunks


# Inject documents directly into a Milvus lite vector database using the Milvus APIs
def inject_documents_milvus(corpus, embedding_model_id, chunk_size_in_tokens):
    collection_name = f"beir_eval_{uuid.uuid4().hex}"

    embedding_model = model.dense.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_id, device="mps"
    )
    embedding_dimension = embedding_model.dim

    db_file = f"./milvus_{uuid.uuid4().hex[0:20]}.db"
    milvus_client = MilvusClient(db_file)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=int(embedding_dimension),
        auto_id=True,
    )

    documents = []
    for doc_id, data in corpus.items():
        full_text = data["title"] + " " + data["text"]
        chunks = llama_stack_style_chunker(full_text, chunk_size_in_tokens)
        for chunk in chunks:
            documents.append(
                {
                    "doc_id": doc_id,
                    "vector": embedding_model.encode_documents([chunk])[0],
                    "text": chunk,
                }
            )

    milvus_client.insert(collection_name=collection_name, data=documents)
    return milvus_client, collection_name, embedding_model


# LlamaStack RAG Retriever
class LlamaStackRAGRetriever:
    def __init__(self, vector_db_id, query_config, top_k=10):
        self.client = llama_stack_client
        self.vector_db_id = vector_db_id
        self.query_config = query_config
        self.top_k = top_k

    def retrieve(self, queries, top_k=None):
        results = {}
        times = {}
        top_k = top_k or self.top_k

        for qid, query in queries.items():
            start_time = time.perf_counter()
            rag_results = self.client.tool_runtime.rag_tool.query(
                vector_db_ids=[self.vector_db_id],
                content=query,
                query_config={**self.query_config, "max_chunks": top_k},
            )
            end_time = time.perf_counter()
            times[qid] = end_time - start_time

            doc_ids = rag_results.metadata.get("document_ids", [])
            scores = {doc_id: 1.0 - (i * 0.01) for i, doc_id in enumerate(doc_ids)}

            results[qid] = scores

        return results, times


class MilvusRetriever:
    def __init__(self, milvus_client, collection_name, embedding_model, top_k=10):
        self.milvus_client = milvus_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, queries, top_k=None):
        results = {}
        times = {}
        top_k = top_k or self.top_k

        for qid, query in queries.items():
            start_time = time.perf_counter()
            data = self.embedding_model.encode_queries([query])
            hits = self.milvus_client.search(
                collection_name=self.collection_name,
                data=data,
                limit=top_k,
                anns_field="vector",
                output_fields=["doc_id"],
            )[0]
            end_time = time.perf_counter()
            times[qid] = end_time - start_time

            doc_ids = [hit["entity"]["doc_id"] for hit in hits]
            scores = {doc_id: 1.0 - (i * 0.01) for i, doc_id in enumerate(doc_ids)}

            results[qid] = scores

        return results, times


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
    """
    Runs permutation_test_for_paired_samples above, prints out the output, and returns true IFF there is a significant difference
    """

    mean_score_a = np.mean(scores_a)
    mean_score_b = np.mean(scores_b)

    p_value = permutation_test_for_paired_samples(scores_a, scores_b)
    print(overview_label)
    print(f" {label_a:<50}: {mean_score_a:>10.4f}")
    print(f" {label_b:<50}: {mean_score_b:>10.4f}")
    print(f" {'p_value':<50}: {p_value:>10.4f}")

    if p_value < 0.05:
        print("  p_value<0.05 so this result is statistically significant")
        # Note that the logic below if wrong if the mean scores are equal, but that can't be true if p<1.
        higher_model_id = label_a if mean_score_a >= mean_score_b else label_b
        print(
            f"  You can conclude that {higher_model_id} generation has a higher score on data of this sort."
        )
        return True
    else:
        import math

        print("  p_value>=0.05 so this result is NOT statistically significant.")
        print(
            "  You can conclude that there is not enough data to tell which is higher."
        )
        num_samples = len(scores_a)
        margin_of_error = 1 / math.sqrt(num_samples)
        print(
            f"  Note that this data includes {num_samples} questions which typically produces a margin of error of around +/-{margin_of_error:.1%}."
        )
        print("  So the two are probably roughly within that margin of error or so.")
        return False


def get_metrics(all_scores):
    for scores_for_dataset in all_scores.values():
        for scores_for_condition in scores_for_dataset.values():
            for scores_for_question in scores_for_condition.values():
                metrics = list(scores_for_question.keys())
                metrics.sort()
                return metrics
    return []


def print_scores(all_scores):
    """
    Iterates through all pairs of scores.  Returns true iff there is a significant differences among any of the pairs.
    """
    metrics = get_metrics(all_scores)
    has_significant_difference = False
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
                is_significant = print_stats_significance(
                    scores_a, scores_b, overview_label, label_a, label_b
                )
                print()
                print()
                if metric != "time":
                    # we exclude time from the has_significant_difference computation because we only want to throw an error if there is a difference in behavior
                    has_significant_difference = (
                        has_significant_difference or is_significant
                    )

    return has_significant_difference


def evaluate_retrieval_with_and_without_llama_stack(
    llama_stack_client,
    datasets,
    vector_db_provider_id,
    embedding_model_id,
    chunk_size_in_tokens=512,
    number_of_search_results=10,
    save_files=False,
):
    all_scores = {}
    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    for dataset_name in datasets:
        all_scores[dataset_name] = {}
        corpus, queries, qrels = load_beir_dataset(dataset_name)

        # Uncomment this line to select only a few documents for debugging
        # corpus = pick_arbitrary_pairs(corpus)

        retrievers = {}

        logger.info(f"Ingesting {dataset_name}, LlamaStackRAGRetriever")
        vector_db_id = inject_documents_llama_stack(
            llama_stack_client,
            corpus,
            vector_db_provider_id,
            embedding_model_id,
            chunk_size_in_tokens,
        )

        # We set max_tokens_in_context=chunk_size_in_tokens*number_of_search_results so that we won't get errors saying that we have too many tokens.
        # These errors don't really matter because we are not generating responses anyway, but they are a distraction.
        query_config = RAGQueryConfig(
            max_chunks=number_of_search_results,
            max_tokens_in_context=chunk_size_in_tokens * number_of_search_results,
        ).model_dump()

        llama_stack_retriever = LlamaStackRAGRetriever(
            vector_db_id, query_config, top_k=number_of_search_results
        )
        retrievers["LlamaStackRAGRetriever"] = llama_stack_retriever

        print(f"Ingesting {dataset_name}, MilvusRetriever")
        milvus_client, collection_name, embedding_model = inject_documents_milvus(
            corpus, embedding_model_id, chunk_size_in_tokens
        )
        milvus_retriever = MilvusRetriever(
            milvus_client,
            collection_name,
            embedding_model,
            top_k=number_of_search_results,
        )
        retrievers["MilvusRetriever"] = milvus_retriever

        for label, retriever in retrievers.items():
            logger.info(f"Retrieving from {label}")
            results, times = retriever.retrieve(queries, top_k=10)

            logger.info("Scoring")
            k_values = [10]

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

            all_scores[dataset_name][label] = scores

            if save_files:
                os.makedirs(results_dir, exist_ok=True)
                util.save_runfile(
                    os.path.join(
                        results_dir, f"{dataset_name}-{vector_db_id}-{label}.run.trec"
                    ),
                    results,
                )
    if save_files:
        logger.info(f"All results in {results_dir}")
    return all_scores


# From Gemini with edits
def pick_arbitrary_pairs(input_dict, num_pairs=5):
    """
    Picks a specified number of arbitrary key-value pairs from a dictionary.

    Args:
      input_dict: The dictionary to pick from.
      num_pairs: The number of key-value pairs to pick. Defaults to 5.

    Returns:
      A new dictionary containing the randomly selected key-value pairs.
      If the input dictionary has fewer items than num_pairs,
      all items from the input dictionary are returned.
    """
    if not isinstance(input_dict, dict):
        raise TypeError("Input must be a dictionary.")
    if not isinstance(num_pairs, int) or num_pairs < 0:
        raise ValueError("Number of pairs must be a non-negative integer.")

    all_items = list(input_dict.items())

    if len(all_items) <= num_pairs:
        return dict(all_items)  # Return all items if fewer than num_pairs

    picked_items = all_items[0:num_pairs]
    return dict(picked_items)


if __name__ == "__main__":
    llama_stack_client = LlamaStackAsLibraryClient("./run.yaml")
    llama_stack_client.initialize()

    all_scores = evaluate_retrieval_with_and_without_llama_stack(
        llama_stack_client,
        DATASETS,
        "milvus",
        "ibm-granite/granite-embedding-125m-english",
    )
    has_significant_difference = print_scores(all_scores)
    if has_significant_difference:
        logger.error(
            "A significant difference was detected.  This is not expected because LlamaStackRAGRetriever and MilvusRetriever are intended to do the same thing.  There might be a new defect of some sort."
        )
        sys.exit(1)
    else:
        logger.info(
            "No significant difference was detected.  This is expected because LlamaStackRAGRetriever and MilvusRetriever are intended to do the same thing.  This result is consistent with everything working as intended."
        )
        sys.exit(0)


# Sample outputs for DATASETS = ["scifact", "fiqa", "arguana"] -- note that this takes much longer because fiqa and arguana are much bigger and take longer to ingest.
"""
scifact map_cut_10
 LlamaStackRAGRetriever                            :     0.6879
 MilvusRetriever                                   :     0.6879
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 300 questions which typically produces a margin of error of around +/-5.8%.
  So the two are probably roughly within that margin of error or so.


scifact ndcg_cut_10
 LlamaStackRAGRetriever                            :     0.7350
 MilvusRetriever                                   :     0.7350
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 300 questions which typically produces a margin of error of around +/-5.8%.
  So the two are probably roughly within that margin of error or so.


scifact time
 LlamaStackRAGRetriever                            :     0.0225
 MilvusRetriever                                   :     0.0173
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that LlamaStackRAGRetriever generation has a higher score on data of this sort.


fiqa map_cut_10
 LlamaStackRAGRetriever                            :     0.3581
 MilvusRetriever                                   :     0.3581
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 648 questions which typically produces a margin of error of around +/-3.9%.
  So the two are probably roughly within that margin of error or so.


fiqa ndcg_cut_10
 LlamaStackRAGRetriever                            :     0.4411
 MilvusRetriever                                   :     0.4411
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 648 questions which typically produces a margin of error of around +/-3.9%.
  So the two are probably roughly within that margin of error or so.


fiqa time
 LlamaStackRAGRetriever                            :     0.0332
 MilvusRetriever                                   :     0.0303
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that LlamaStackRAGRetriever generation has a higher score on data of this sort.


/Users/bmurdock/beir/beir-venv-310/lib/python3.10/site-packages/scipy/stats/_resampling.py:1492: RuntimeWarning: overflow encountered in scalar power
  n_max = factorial(n_obs_sample)**n_samples
arguana map_cut_10
 LlamaStackRAGRetriever                            :     0.2927
 MilvusRetriever                                   :     0.2927
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 1406 questions which typically produces a margin of error of around +/-2.7%.
  So the two are probably roughly within that margin of error or so.


arguana ndcg_cut_10
 LlamaStackRAGRetriever                            :     0.4251
 MilvusRetriever                                   :     0.4251
 p_value                                           :     1.0000
  p_value>=0.05 so this result is NOT statistically significant.
  You can conclude that there is not enough data to tell which is higher.
  Note that this data includes 1406 questions which typically produces a margin of error of around +/-2.7%.
  So the two are probably roughly within that margin of error or so.


arguana time
 LlamaStackRAGRetriever                            :     0.0303
 MilvusRetriever                                   :     0.0239
 p_value                                           :     0.0002
  p_value<0.05 so this result is statistically significant
  You can conclude that LlamaStackRAGRetriever generation has a higher score on data of this sort.
"""
