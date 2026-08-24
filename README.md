# Reinforcement Learning for Query-Focused Context Selection

## Baseline Branch — BM25 Retrieval

This branch establishes the first retrieval baseline for the project using the QASPER-RAG dataset and BM25.

The purpose of this branch is to create a reproducible reference point against which later retrieval and context-selection approaches can be compared.

The current system performs **retrieval only**. No answer generation, model fine-tuning, or reinforcement learning is performed in this branch.

---

# 1. Objective

The overall project investigates query-focused context selection from long textual contexts using progressively more sophisticated retrieval and selection methods.

The experimental progression will eventually be:

```text
BM25
  ↓
Dense Retrieval
  ↓
Hybrid Retrieval
  ↓
Reranking
  ↓
LLM-based Context Selection
  ↓
RL-based Context Selection
````

This branch implements the first stage:

```text
Question
   ↓
BM25
   ↓
Ranked Corpus Chunks
   ↓
QREL Evaluation
   ↓
Retrieval Metrics
```

The BM25 system serves as the initial lexical retrieval baseline.

---

# 2. Dataset

The benchmark used in this branch is **QASPER-RAG**.

The dataset contains:

* 1,585 research papers
* 81,550 corpus chunks
* 1,310 test questions
* 3,434 test relevance judgments

The corpus is already segmented into retrieval chunks. Therefore, this branch does not perform additional document chunking.

## Dataset structure

```text
data/
└── qasper/
    ├── corpus/
    │   └── corpus.parquet
    │
    ├── queries/
    │   └── test.parquet
    │
    ├── qrels/
    │   └── test.parquet
    │
    └── answers/
        └── test.parquet
```

---

# 3. Dataset Components

## 3.1 Corpus

File:

```text
data/qasper/corpus/corpus.parquet
```

Size:

```text
81,550 chunks
```

Columns:

```text
id
title
section_name
text
```

Each row represents a pre-existing retrieval chunk.

No additional chunking is performed by the BM25 baseline.

---

## 3.2 Test Queries

File:

```text
data/qasper/queries/test.parquet
```

Size:

```text
1,310 questions
```

Columns:

```text
id
text
```

Each query represents a question used for retrieval evaluation.

---

## 3.3 Qrels

File:

```text
data/qasper/qrels/test.parquet
```

Size:

```text
3,434 relevance judgments
```

Columns:

```text
query_id
corpus_id
score
```

The qrels provide the retrieval ground truth.

All observed relevance scores are:

```text
score = 1
```

Therefore, the relevance judgments are binary.

A qrel indicates that a particular corpus chunk is relevant to a particular query.

For example:

```text
query_id     corpus_id       score
------------------------------------------------
q123         chunk_001       1
q123         chunk_017       1
q123         chunk_084       1
```

means that three corpus chunks are annotated as relevant to `q123`.

---

## 3.4 Answers

File:

```text
data/qasper/answers/test.parquet
```

This file contains reference answers.

It is **not used by the BM25 retrieval evaluation**.

Answer generation and answer-quality evaluation will be introduced in later branches when an LLM-based generation component is added.

---

# 4. Dataset Validation

Before implementing BM25, the dataset was inspected to verify its structure.

The following checks were performed.

### Corpus

```text
Corpus chunks: 81,550
Unique papers: 1,585
Unique sections: 13,543
```

### Queries

```text
Test queries: 1,310
```

All test queries have corresponding relevance judgments.

```text
Queries with qrels: 1,310
Queries without qrels: 0
```

### Qrels

```text
Total qrels: 3,434
```

All qrel corpus IDs exist in the corpus.

```text
Qrel corpus IDs found in corpus: 2,960
Qrel corpus IDs NOT found in corpus: 0
```

There are, on average:

```text
3,434 / 1,310 = 2.62
```

relevant chunks per question.

The number of relevant chunks varies considerably:

```text
Minimum: 1
Maximum: 67
Mean:    2.62
Median:  2
```

This variation is important for future context-selection and reinforcement-learning experiments because different questions require different amounts of relevant evidence.

---

# 5. Retrieval Setup

The BM25 implementation uses:

```text
rank_bm25
```

with:

```text
BM25Okapi
```

The corpus tokenizer is currently:

```text
lowercase
+
whitespace splitting
```

No stemming, lemmatization, stop-word removal, or additional linguistic preprocessing is applied in this baseline.

The same preprocessing is applied to queries.

---

# 6. Retrieval Scope

The current baseline performs **global retrieval**.

For every query, BM25 searches the complete corpus:

```text
Query
  ↓
81,550 corpus chunks
  ↓
BM25 scoring
  ↓
ranked results
```

The system does not restrict retrieval to the source paper.

This establishes a global-corpus retrieval baseline.

A paper-scoped retrieval experiment may be investigated separately as a diagnostic experiment, but it is not part of this baseline.

---

# 7. Evaluation

The retrieval output is compared against:

```text
data/qasper/qrels/test.parquet
```

The following metrics are currently calculated:

* Precision@K
* Recall@K
* F1@K
* nDCG@K
* MRR

The evaluated values of K are:

```text
K = 1, 3, 5, 10, 20, 50
```

Latency is also measured.

---

# 8. Metric Definitions

## Recall@K

Recall@K measures the fraction of relevant corpus chunks retrieved within the top K results.

```text
Recall@K =
relevant chunks retrieved in top K
----------------------------------
total relevant chunks
```

Recall is particularly important for the eventual context-selection component because missing evidence can prevent a downstream system from answering a question correctly.

---

## Precision@K

Precision@K measures the fraction of the top K retrieved chunks that are relevant.

```text
Precision@K =
relevant chunks retrieved in top K
----------------------------------
K
```

---

## F1@K

F1 combines Precision@K and Recall@K using their harmonic mean.

```text
F1 = 2PR / (P + R)
```

---

## nDCG@K

nDCG evaluates the ranking quality while giving higher importance to relevant chunks appearing near the top of the ranking.

---

## MRR

Mean Reciprocal Rank measures the reciprocal rank of the first relevant retrieved chunk, averaged over all queries.

---

# 9. BM25 Baseline Results

The current BM25 baseline was executed over:

```text
81,550 corpus chunks
1,310 test questions
3,434 qrels
```

## Retrieval results

|  K | Precision@K | Recall@K |   F1@K | nDCG@K |
| -: | ----------: | -------: | -----: | -----: |
|  1 |      0.0359 |   0.0186 | 0.0224 | 0.0359 |
|  3 |      0.0275 |   0.0415 | 0.0301 | 0.0404 |
|  5 |      0.0211 |   0.0541 | 0.0278 | 0.0445 |
| 10 |      0.0152 |   0.0765 | 0.0236 | 0.0524 |
| 20 |      0.0098 |   0.0972 | 0.0171 | 0.0587 |
| 50 |      0.0055 |   0.1283 | 0.0102 | 0.0665 |

### MRR

```text
0.0644
```

---

# 10. Latency

The latest BM25 run produced:

```text
Mean latency:   271.50 ms/query
Median latency: 252.96 ms/query
P95 latency:    477.73 ms/query
```

The BM25 index construction time was approximately:

```text
3.032 seconds
```

Index construction time and query latency are reported separately.

The current implementation calculates BM25 scores over the complete corpus and then sorts the resulting document scores.

Therefore, these latency values represent the performance of this Python-based implementation and should not be interpreted as the theoretical latency of an optimized production BM25 engine.

---

# 11. Output

The per-query evaluation results are saved to:

```text
data/qasper/bm25_results.csv
```

The output contains query-level retrieval metrics and latency measurements.

This allows later experiments to compare systems on the same individual queries rather than only comparing aggregate metrics.

---

# 12. Running the Baseline

Install the required packages:

```bash
pip install datasets pandas pyarrow tqdm rank-bm25 numpy
```

The dataset preparation script creates the local QASPER files.

Run:

```bash
python prepare_qasper.py
```

The BM25 baseline can then be executed with:

```bash
python bm25_baseline.py
```

Optional dataset inspection:

```bash
python inspect_qasper.py
```

---

# 13. Current Project Files

The current branch contains:

```text
.
├── README.md
├── prepare_qasper.py
├── inspect_qasper.py
├── bm25_baseline.py
│
└── data/
    └── qasper/
        ├── corpus/
        │   └── corpus.parquet
        │
        ├── queries/
        │   └── test.parquet
        │
        ├── qrels/
        │   └── test.parquet
        │
        ├── answers/
        │   └── test.parquet
        │
        └── bm25_results.csv
```

---

# 14. Experimental Principle

All future retrieval systems should use the same:

```text
Test queries
Qrels
Evaluation metrics
K values
```

This allows direct comparison between retrieval methods.

The planned comparison is:

```text
                    QASPER TEST SET
                          │
          ┌───────────────┼───────────────┐
          │               │               │
        Queries          Corpus          Qrels
          │               │               │
          └───────────────┼───────────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
           BM25                   Dense Retrieval
             │                         │
             └────────────┬────────────┘
                          │
                       Hybrid
                          │
                       Reranker
                          │
                Context Selection
                          │
                   RL-based Agent
```

---

# 15. Scope of This Branch

This branch intentionally does **not** include:

* Dense embeddings
* Vector databases
* Hybrid retrieval
* Cross-encoder reranking
* LLM answer generation
* Reinforcement learning
* Policy training
* Reward modeling
* Fine-tuning

Those components will be implemented in subsequent experimental branches.

The purpose of this branch is to establish a clean and reproducible **lexical retrieval baseline**.

---

# 16. Baseline Status

```text
Dataset acquisition          COMPLETE
Dataset validation           COMPLETE
Corpus preparation           COMPLETE
Test split preparation       COMPLETE
Qrel validation              COMPLETE
BM25 implementation          COMPLETE
Retrieval evaluation         COMPLETE
Latency measurement          COMPLETE
Per-query results            COMPLETE

BM25 baseline                COMPLETE
```

The current BM25 baseline is therefore the reference point for the next retrieval experiment: **dense semantic retrieval**.

```

This is the version I'd commit with the BM25 branch. It documents **what we actually ran**, rather than mixing in the future RL architecture prematurely.
```
