from pathlib import Path
import time

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

BASE = Path("data/qasper")

CORPUS_PATH = BASE / "corpus" / "corpus.parquet"
QUERIES_PATH = BASE / "queries" / "test.parquet"
QRELS_PATH = BASE / "qrels" / "test.parquet"


K_VALUES = [1, 3, 5, 10, 20, 50]

def tokenize(text):
    return text.lower().split()

print("Loading corpus...")
corpus = pd.read_parquet(CORPUS_PATH)

print("Loading test queries...")
queries = pd.read_parquet(QUERIES_PATH)

print("Loading qrels...")
qrels = pd.read_parquet(QRELS_PATH)

print()
print("Corpus:", len(corpus))
print("Queries:", len(queries))
print("Qrels:", len(qrels))
print()

# Make sure IDs are strings
corpus["id"] = corpus["id"].astype(str)
queries["id"] = queries["id"].astype(str)
qrels["query_id"] = qrels["query_id"].astype(str)
qrels["corpus_id"] = qrels["corpus_id"].astype(str)

print("Tokenizing corpus...")

tokenized_corpus = [
    tokenize(text)
    for text in tqdm(corpus["text"].fillna(""), desc="Tokenizing")
]

print("Building BM25 index...")

index_start = time.perf_counter()

bm25 = BM25Okapi(tokenized_corpus)

index_time = time.perf_counter() - index_start

print(f"BM25 index built in {index_time:.3f} seconds")
print()

# query_id -> set of relevant corpus IDs
qrels_dict = (
    qrels.groupby("query_id")["corpus_id"]
    .apply(set)
    .to_dict()
)

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    if not retrieved_k:
        return 0.0

    hits = sum(doc_id in relevant for doc_id in retrieved_k)

    return hits / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]

    hits = sum(doc_id in relevant for doc_id in retrieved_k)

    return hits / len(relevant)


def f1_at_k(retrieved, relevant, k):
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def reciprocal_rank(retrieved, relevant):
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]

    dcg = 0.0

    for rank, doc_id in enumerate(retrieved_k, start=1):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg

corpus_ids = corpus["id"].tolist()

results = []

print("Running BM25 retrieval...")

for _, row in tqdm(
    queries.iterrows(),
    total=len(queries),
    desc="Queries"
):

    query_id = str(row["id"])
    query_text = str(row["text"])

    relevant = qrels_dict.get(query_id, set())

    query_tokens = tokenize(query_text)

    start = time.perf_counter()

    scores = bm25.get_scores(query_tokens)

    # Get ranking
    ranked_indices = np.argsort(scores)[::-1]

    retrieval_time = time.perf_counter() - start

    ranked_doc_ids = [
        corpus_ids[i]
        for i in ranked_indices
    ]

    result = {
        "query_id": query_id,
        "latency_ms": retrieval_time * 1000,
        "num_relevant": len(relevant),
    }

    for k in K_VALUES:
        result[f"precision@{k}"] = precision_at_k(
            ranked_doc_ids,
            relevant,
            k
        )

        result[f"recall@{k}"] = recall_at_k(
            ranked_doc_ids,
            relevant,
            k
        )

        result[f"f1@{k}"] = f1_at_k(
            ranked_doc_ids,
            relevant,
            k
        )

        result[f"ndcg@{k}"] = ndcg_at_k(
            ranked_doc_ids,
            relevant,
            k
        )

    result["mrr"] = reciprocal_rank(
        ranked_doc_ids,
        relevant
    )

    results.append(result)

results_df = pd.DataFrame(results)

print()
print("=" * 70)
print("BM25 RESULTS")
print("=" * 70)

for k in K_VALUES:

    print(f"\nK = {k}")

    print(
        f"Precision@{k}: "
        f"{results_df[f'precision@{k}'].mean():.4f}"
    )

    print(
        f"Recall@{k}:    "
        f"{results_df[f'recall@{k}'].mean():.4f}"
    )

    print(
        f"F1@{k}:        "
        f"{results_df[f'f1@{k}'].mean():.4f}"
    )

    print(
        f"nDCG@{k}:      "
        f"{results_df[f'ndcg@{k}'].mean():.4f}"
    )


print()
print(f"MRR: {results_df['mrr'].mean():.4f}")

print()
print(
    f"Mean latency: "
    f"{results_df['latency_ms'].mean():.2f} ms"
)

print(
    f"Median latency: "
    f"{results_df['latency_ms'].median():.2f} ms"
)

print(
    f"P95 latency: "
    f"{results_df['latency_ms'].quantile(0.95):.2f} ms"
)

OUTPUT_PATH = BASE / "bm25_results.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print(f"Saved results to: {OUTPUT_PATH}")