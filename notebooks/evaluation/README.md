# README: RAG System Evaluation Framework

This repository provides a set of tools to evaluate and compare the performance of Retrieval-Augmented Generation (RAG) systems. Specifically, these notebooks demonstrate a framework for:

1.  **Synthesizing a question-answer dataset** from a source document.
2.  **Evaluating two RAG pipelines** (Llama Stack and LlamaIndex) on the generated dataset.
3.  **Analyzing the results** using statistical methods to determine significance.

The primary goal is to offer a reproducible methodology for comparing RAG system performance on a given knowledge base.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Summary of Findings](#summary-of-findings)
- [Detailed Results](#detailed-results)
- [Key Limitations of this Study](#key-limitations-of-this-study)
- [Further Observations](#further-observations)

## Project Structure

This directory includes the following components:

- **Jupyter Notebooks**:
    * [`make-sample-questions.ipynb`](./make-sample-questions.ipynb): Generates a dataset of sample questions and reference answers from a source document.
    * [`evaluate-using-sample-questions-lls-vs-li.ipynb`](./evaluate-using-sample-questions-lls-vs-li.ipynb): Runs Llama Stack and LlamaIndex RAG pipelines on the generated questions, evaluates their responses using the Ragas framework, and performs statistical significance testing with SciPy.

- **Supporting Code**:
    * [`evaluation_utilities.py`](./evaluation_utilities.py): Utility functions and helper code for the evaluation notebooks.

- **Sample Data**:
    * [`qna-ibm-2024-2250-2239.json`](./qna-ibm-2024-2250-2239.json): A Q&A dataset generated from the IBM 2024 annual report without special instructions.
    * [`qna-ibm-2024b-2220-2196.json`](./qna-ibm-2024b-2220-2196.json): A Q&A dataset generated from the same report, but using the default special instructions in the notebook to produce more diverse questions.
    * **Note on filenames**: The numbers in the JSON filenames (`{configured_questions}-{final_question_count}`) may not perfectly match the final counts in the file due to de-duplication steps.

- **Configuration**:
    * [`requirements.txt`](./requirements.txt): A list of Python libraries required to run the notebooks.
    * [`run.yaml`](./run.yaml): A configuration file for the Llama Stack server.

## Getting Started

Follow these steps to reproduce the evaluation.

### 1. Install Dependencies

Install all the necessary Python libraries using pip:

```bash
pip install -r requirements.txt
```

### 2. Start the Llama Stack Server

The evaluation notebook requires a running Llama Stack server. Start it from your command line using the provided configuration:

```bash
llama stack run run.yaml --image-type venv
```

### 3. Run the Notebooks

1. **(Optional)** Run `make-sample-questions.ipynb` if you want to generate your own question-answer dataset from a new document.
2. Run `evaluate-using-sample-questions-lls-vs-li.ipynb` to execute the comparison between Llama Stack and LlamaIndex using one of the sample `.json` files.

> **Note on Scale**: Both notebooks are configured by default to run on a limited number of questions for quick results. Instructions are included within the notebooks on how to adjust the configuration to run on the full datasets.

## Summary of Findings

Across both datasets, our results show:

- **Higher Accuracy for Llama Stack**: Llama Stack consistently achieved a small but statistically significant advantage in accuracy metrics (`nv_accuracy` and `domain_specific_rubrics`) for questions that had reference answers.
- **Superior Handling of Unanswerable Questions**: Llama Stack demonstrated a much stronger ability to correctly identify and refuse to answer questions that were designed to be unanswerable based on the source document. A higher "Percent Unanswered" score is better in this context.

We hypothesize these differences may stem from variations in model prompting, document chunking strategies, or text processing between the two frameworks.

## Detailed Results

The tables below summarize the performance metrics from our full runs. All p-values are less than `0.05`, indicating the observed differences are statistically significant.

### Dataset 1: `qna-ibm-2024-2250-2239.json`

| Metric (Higher is Better) | Llama Stack (`gpt-3.5-turbo`) | LlamaIndex (`gpt-3.5-turbo`) | p-value | Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **Questions with Answers (1479)** | | | | |
| `nv_accuracy` | 0.5046 | 0.4696 | 0.0002 | Advantage for Llama Stack |
| `domain_specific_rubrics` (score out of 5) | 3.9757 | 3.9033 | 0.0310 | Advantage for Llama Stack |
| **Questions without Answers (760)** | | | | |
| `Percent Unanswered` | **23.95%** | 8.42% | 0.0002 | Advantage for Llama Stack |

### Dataset 2: `qna-ibm-2024b-2220-2196.json`

| Metric (Higher is Better) | Llama Stack (`gpt-3.5-turbo`) | LlamaIndex (`gpt-3.5-turbo`) | p-value | Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **Questions with Answers (1402)** | | | | |
| `nv_accuracy` | 0.4918 | 0.4358 | 0.0002 | Advantage for Llama Stack |
| `domain_specific_rubrics` (score out of 5) | 3.9073 | 3.7582 | 0.0002 | Advantage for Llama Stack |
| **Questions without Answers (794)** | | | | |
| `Percent Unanswered` | **31.74%** | 7.68% | 0.0002 | Advantage for Llama Stack |

## Key Limitations of this Study

While these results are informative, it is crucial to consider their limitations:

1. **Single Dataset**: This evaluation uses only one document. Performance could vary significantly with different data types, topics, or multiple documents.
2. **Synthetic Questions**: Questions generated by an LLM may not perfectly represent the questions real users would ask. While we used prompt engineering to increase diversity, it is not a substitute for real-world query logs.
3. **Imperfect Ground Truth**: Our reference answers were generated by a powerful RAG system (using `gpt-4o`), not by humans. This introduces noise into the evaluation, though we assume it affects both systems equally.
4. **Assumption on Unanswerable Questions**: We assume that if our reference RAG system doesn't answer a question, it is truly unanswerable. This assumption may be flawed and could contribute to the low scores for refusing to answer.
5. **Potential for Framework Bias**: Since the reference RAG system was built with LlamaIndex, it could theoretically introduce a bias in favor of LlamaIndex. However, the results show Llama Stack outperforming, suggesting any such bias is likely minimal.
6. **Evaluation Metric Imperfections**: The Ragas metrics and the `gpt-4o` model used to power them are not perfect. This is another source of potential noise.
7. **Custom Metric Validity**: The custom prompt used to determine if a question was answered has not been rigorously validated, though it appears to function well upon casual inspection.

## Further Observations

A key takeaway is that the **absolute performance of both RAG systems is quite low** in this challenging evaluation. Accuracy hovers around 50%, and the ability to correctly ignore unanswerable questions is even lower.

We believe this is partly due to the limitations mentioned above, but also because our question generation method produces a more difficult and diverse set of questions than standard benchmarks. Future work should validate whether these challenging synthetic questions are more representative of the difficulties a RAG system would face in a real-world deployment.