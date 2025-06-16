from ragas.evaluation import evaluate
from ragas import EvaluationDataset

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import ChatMessage

import numpy as np
import pandas as pd
import itertools

import os
import json
import re

from pathlib import Path
import copy
import time
import sys


# This is the default RAG prompt template for Llama Index
PROMPT_TEMPLATE="Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: "


# Assisted by watsonx Code Assistant 
def string_to_int(s):
    try:
        return int(s)
    except ValueError:
        return None

def make_simple_index(file_paths, embed_model):
    reader = DoclingReader()
    node_parser = MarkdownNodeParser()

    index = VectorStoreIndex.from_documents(
        documents=reader.load_data(file_path=file_paths),
        transformations=[node_parser],
        embed_model=embed_model,

    )
    return index


# For details on the APIs used, see https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/usage_pattern/#low-level-composition-api
def run_rag(qna, generator_model, idx, number_of_search_results=5):
    dataset = copy.deepcopy(qna)

    retriever = VectorIndexRetriever(
        index=idx,
        similarity_top_k=number_of_search_results,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer(llm = generator_model)
    )

    for entry in dataset:
        question = entry['user_input']
        result = query_engine.query(question)
        entry["response"] = result.response.strip()
        entry["retrieved_contexts"] = [n.text for n in result.source_nodes]
    return dataset


def run_reference_rag(qna, generator_model, idx, number_of_search_results=20, number_of_selected_results=5, minimum_score_for_a_reference_context=7):
    dataset = copy.deepcopy(qna)

    retriever = VectorIndexRetriever(
        index=idx,
        similarity_top_k=number_of_search_results,
    )

    for entry in dataset:
        question = entry["user_input"]

        nodes  = retriever.retrieve(question)
        reference_search_results = []
        for node in nodes:
            message_content = f"Consider the following question and answer.\n question: {question}\n passage:\n----------\n{node.text}.\n----------\n. Rate the usefulness of this passage for answering the question on a scale of 1 to 10 where 1 is totally useless and 10 is completely answers the question.  Please respond only with an integer from 1 to 10."
            messages = [ChatMessage(role="user", content=message_content)]
            response = generator_model.chat(messages)
            text = response.message.blocks[0].text.strip().lower()
            response = text
            score = string_to_int(response)
            if score is None:
                score = 3.5 # rank something that didn't get a valid score as better than something we know is very bad but worse than anything good
                # Note that this 3.5 value only matters if minimum_score_for_a_reference_answer is below that value.  Otherwise, it will get discarded anyway.
            reference_search_results.append((node.text, score))

        # Sort from best to worst
        reference_search_results.sort(key=lambda x: -x[1])

        # Pick the best ones
        top_results = reference_search_results[0:number_of_selected_results]

        # Now eliminate any that are below the threshold for a reference context
        selected_results = [r for r in top_results if r[1] >= minimum_score_for_a_reference_context]

        # Then sort from worst to best so the best ones are closer to that question.  (Some people think that helps)
        selected_results.sort(key=lambda x: x[1])
        reference_contexts = [selected_result[0] for selected_result in selected_results]
        context_str = "\n-------\n".join(reference_contexts)

        message_text = PROMPT_TEMPLATE.format(**{"context_str":context_str, "query_str": question})
        messages = [ChatMessage(role="user", content=message_text)]
        response = generator_model.chat(messages)

        entry["reference"] = response.message.blocks[0].text.strip()
        entry["reference_contexts"] = reference_contexts
    return dataset


# Assisted by watsonx Code Assistant 
def list_files(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for file in os.listdir(path):
        file_path = Path(os.path.join(path, file))
        if os.path.isfile(file_path):
            files.append(file_path)
    return files


# Assisted by watsonx Code Assistant 
def write_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent = 4)


# Assisted by watsonx Code Assistant 
def read_json(filename):
    with open(filename, "r") as f:
        data_loaded = json.load(f)
        return data_loaded


def mean_diff_paired(x, y):
    return np.mean(x) - np.mean(y)


# Rubrics from https://github.com/instructlab/eval/blob/main/src/instructlab/eval/ragas.py which got them from ragas v0.2.11
# and has them "hardcoded in case ragas makes any changes to their DEFAULT_WITH_REFERENCE_RUBRICS in the future".
SCORING_RUBRICS = {
    "score1_description": "The response is entirely incorrect, irrelevant, or does not align with the reference in any meaningful way.",
    "score2_description": "The response partially matches the reference but contains major errors, significant omissions, or irrelevant information.",
    "score3_description": "The response aligns with the reference overall but lacks sufficient detail, clarity, or contains minor inaccuracies.",
    "score4_description": "The response is mostly accurate, aligns closely with the reference, and contains only minor issues or omissions.",
    "score5_description": "The response is fully accurate, completely aligns with the reference, and is clear, thorough, and detailed.",
}

def run_ragas(datasets, evaluator_llm_for_ragas, metrics):

    results = {}
    for label, data in datasets.items():
        evaluation_dataset = EvaluationDataset.from_list(data)
        result = evaluate(
            metrics=metrics,
            batch_size=4,
            dataset=evaluation_dataset,
            llm=evaluator_llm_for_ragas,
            show_progress=True
        )
        results[label] = result
    return results

def mean_difference_paired(x, y):
    return np.mean(x - y)


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
        print(f"  You can conclude that {higher_model_id} generation is better on data of this sort")
        return True, p_value, mean_score_a, mean_score_b
    else:
        import math

        print("  p_value>=0.05 so this result is NOT statistically significant.")
        print("  You can conclude that there is not enough data to tell which is better.")
        num_samples = len(scores_a)
        margin_of_error = 1 / math.sqrt(num_samples)
        print(
            f"  Note that this data includes {num_samples} questions which typically produces a margin of error of around +/-{margin_of_error:.1%}."
        )
        print("  So the two are probably roughly within that margin of error or so.")
        return False, p_value, mean_score_a, mean_score_b
    

def report_results_with_significance(results, metrics, subset_of_rows = None):
    """
    Iterates through all pairs of results for all metrics and computes the mean value of the metrics and whether the results are signifant.
    If subset_of_rows is set, then only that subset of all the rows are used.  (For example, this can be used to report results for just
    the subset of questions for which there is at least one reference context).
    """

    result_pairs = list(itertools.combinations(results.keys(), 2))
    result_summary = []
    for result_pair in result_pairs:
        result_summary_for_metric = {}
        for metric in metrics:
            results0 = results[result_pair[0]].to_pandas()
            results1 = results[result_pair[1]].to_pandas()

            if subset_of_rows:
                results0 = results0.iloc[subset_of_rows]
                results1 = results1.iloc[subset_of_rows]

            group0 = results0[metric.name].copy()
            group1 = results1[metric.name].copy()
            # Treat all NaN values as 0.
            # This is important because NaN breaks our significance testing and is common in some Ragas metrics such as Faithfulness.
            group0[np.isnan(group0)] = 0
            group1[np.isnan(group1)] = 0

            overview_label = f"{result_pair[0]} {result_pair[1]} {metric.name}"
            _, p_value, score0, score1 = print_stats_significance(group0, group1, overview_label, result_pair[0], result_pair[1])

            result_summary_for_metric[metric.name] = {
                result_pair[0]: float(score0),
                result_pair[1]: float(score1),
                "p": p_value
            }
        result_summary.append(result_summary_for_metric)
    return result_summary


def convert_to_dataframe(result_summary_list):
    rows = []
    for result_summary_row in result_summary_list:
        for metric, models in result_summary_row.items():
            row = {'Metric': metric}
            for label, value in models.items():
                row[label] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    
    return df


def clean_label_for_excel(s):
    # Regex from Google Gemini
    pattern = r"[: \[\]\*?/\\]"
    # Note the limit of 31 characters, also for Excel
    return re.sub(pattern, "_", s[:31])


def write_df_to_workbook(df, writer, sheet_name, number_format):
    sheet_name = clean_label_for_excel(sheet_name)
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    worksheet = writer.sheets[sheet_name]

    for i, col in enumerate(df.columns):
        max_len = min(100, max(df[col].astype(str).apply(len).max(), len(col)) + 2)
        worksheet.set_column(i, i, max_len)
        if df[col].dtype == 'float64':
            worksheet.set_column(i, i, max_len, number_format)



# Adapted from https://github.com/rh-aiservices-bu/rhel-ai-poc/blob/main/eval/eval_utils.py
def write_excel(results, result_summary_all_rows, result_summary_rows_with_complete_reference_answers, output_file):
    with pd.ExcelWriter(output_file) as writer:
        workbook = writer.book
        number_format = workbook.add_format({'num_format': '0.0000'})

        if result_summary_all_rows is not None:
            write_df_to_workbook(convert_to_dataframe(result_summary_all_rows), writer, "all_rows", number_format)
        if result_summary_rows_with_complete_reference_answers is not None:
            write_df_to_workbook(convert_to_dataframe(result_summary_rows_with_complete_reference_answers), writer, "rows_with_complete_answers", number_format)
        
        for label, ragas_result in results.items():
            write_df_to_workbook(ragas_result.to_pandas(), writer, label, number_format)


# Some code to run something checking for timeout and protocol errors and then retrying.  This is important if you are using
# a hosted service such as watsonx.ai to provide the model in case the servers are having issues.

def run_with_retries(lambda_to_run, max_retries, delay_between_retries_seconds):
    retries = 0
    while True:
        try:
            return lambda_to_run()
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                print(f"An exception of type {e.__class__.__name__} occurred.  Waiting {delay_between_retries_seconds} seconds to give the servers time to stabilize and then resuming.", file=sys.stderr)
                # Wait 30 seconds because if there is a transient issue on the model service, then we want to give it time to clear.
                time.sleep(delay_between_retries_seconds)
            else:
                print(f"An exception of type {e.__class__.__name__} occurred.  Maximum number of retries exceeded.  Terminating.", file=sys.stderr)
                raise e