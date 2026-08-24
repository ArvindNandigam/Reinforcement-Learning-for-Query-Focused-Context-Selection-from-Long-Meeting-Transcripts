from datasets import load_dataset
from pathlib import Path

BASE = Path("data/qasper")

(BASE / "corpus").mkdir(parents=True, exist_ok=True)
(BASE / "queries").mkdir(parents=True, exist_ok=True)
(BASE / "qrels").mkdir(parents=True, exist_ok=True)
(BASE / "answers").mkdir(parents=True, exist_ok=True)

print("Loading corpus...")
corpus = load_dataset(
    "DinoStackAI/qasper-rag",
    "corpus",
    split="train"
)

print("Loading test queries...")
queries = load_dataset(
    "DinoStackAI/qasper-rag",
    "queries",
    split="test"
)

print("Loading test qrels...")
qrels = load_dataset(
    "DinoStackAI/qasper-rag",
    "qrels",
    split="test"
)

print("Loading test answers...")
answers = load_dataset(
    "DinoStackAI/qasper-rag",
    "answers",
    split="test"
)

print("\nDataset sizes:")
print(f"Corpus:  {len(corpus)}")
print(f"Queries: {len(queries)}")
print(f"Qrels:   {len(qrels)}")
print(f"Answers: {len(answers)}")

print("\nSaving...")

corpus.to_parquet(BASE / "corpus" / "corpus.parquet")
queries.to_parquet(BASE / "queries" / "test.parquet")
qrels.to_parquet(BASE / "qrels" / "test.parquet")
answers.to_parquet(BASE / "answers" / "test.parquet")

print("\nDone.")

print("\nFiles:")
for path in BASE.rglob("*"):
    if path.is_file():
        print(f"{path}  ({path.stat().st_size / 1024 / 1024:.2f} MB)")