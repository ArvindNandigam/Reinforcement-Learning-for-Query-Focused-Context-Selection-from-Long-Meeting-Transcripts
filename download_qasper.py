from datasets import load_dataset
from pathlib import Path


BASE = Path("data/qasper")

(BASE / "corpus").mkdir(parents=True, exist_ok=True)
(BASE / "queries").mkdir(parents=True, exist_ok=True)
(BASE / "qrels").mkdir(parents=True, exist_ok=True)
(BASE / "answers").mkdir(parents=True, exist_ok=True)


print("Downloading corpus...")
corpus = load_dataset(
    "DinoStackAI/qasper-rag",
    "corpus",
    split="train"
)

print("Downloading test queries...")
queries = load_dataset(
    "DinoStackAI/qasper-rag",
    "queries",
    split="test"
)

print("Downloading test qrels...")
qrels = load_dataset(
    "DinoStackAI/qasper-rag",
    "qrels",
    split="test"
)

print("Downloading test answers...")
answers = load_dataset(
    "DinoStackAI/qasper-rag",
    "answers",
    split="test"
)


print("\nDataset sizes")
print("------------------------------")
print(f"Corpus : {len(corpus):,}")
print(f"Queries: {len(queries):,}")
print(f"Qrels  : {len(qrels):,}")
print(f"Answers: {len(answers):,}")


print("\nSaving local copies...")


corpus.to_parquet(
    BASE / "corpus" / "corpus.parquet"
)

queries.to_parquet(
    BASE / "queries" / "test.parquet"
)

qrels.to_parquet(
    BASE / "qrels" / "test.parquet"
)

answers.to_parquet(
    BASE / "answers" / "test.parquet"
)


print("\nDownload complete.")