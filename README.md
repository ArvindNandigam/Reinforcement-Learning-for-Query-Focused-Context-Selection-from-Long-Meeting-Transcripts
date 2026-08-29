# Reinforcement Learning for Query-Focused Context Selection from Long Meeting Transcripts

An adaptive retrieval system for long meeting transcripts that selects relevant evidence under token constraints. This project compares multiple retrieval baselines (BM25, dense retrieval, reranking, sequential methods) and uses reinforcement learning to optimize evidence selection for accuracy and context efficiency.

## Overview

The goal is to develop an intelligent retrieval pipeline that can:
- Select the most relevant passages from long meeting transcripts to answer specific queries
- Operate under token budget constraints
- Balance retrieval accuracy with context efficiency
- Learn optimal selection policies using reinforcement learning

## Project Structure

```
├── baseline2-qasper-dense-retrieval.ipynb  # Dense retrieval baseline implementation
├── download_qasper.py                       # Script to download QASPER dataset
├── results/
│   └── ALL_EXPERIMENTS.csv                 # Comprehensive experimental results
├── data/
│   └── qasper/
│       ├── corpus/                          # Document corpus
│       ├── queries/                         # Test queries
│       ├── qrels/                          # Query-document relevance judgments
│       └── answers/                        # Reference answers
└── rl_rag_project/                          # Main project package
```

## Dataset: QASPER

This project uses the **QASPER** (Question Answering and Summarization Evaluation Research) dataset:
- **Corpus**: Academic papers with 3,158 documents
- **Test Queries**: 660 queries
- **Test Qrels**: Relevance judgments for query-document pairs
- **Test Answers**: Reference answers for evaluation

### Download Dataset

```bash
python download_qasper.py
```

## Retrieval Methods

### Implemented Baselines

The project evaluates multiple retrieval approaches:

1. **Dense Retrieval**
   - Uses pre-trained embedding models (E5, BGE)
   - Supports multiple indexing methods:
     - **IVF-Flat**: Inverted File with flat quantization
     - **IVF-PQ**: Inverted File with Product Quantization
     - **HNSW**: Hierarchical Navigable Small World (approximate nearest neighbor search)

2. **Embedding Models**
   - **E5** (`intfloat/e5-base-v2`): 768-dimensional embeddings
   - **BGE** (`BAAI/bge-base-en-v1.5`): 768-dimensional embeddings

3. **Index Configurations**
   - Various parameters for memory-efficiency and speed trade-offs
   - Configurable M, efConstruction, efSearch (HNSW)
   - Configurable nlist, nprobe (IVF)
   - PQ parameters for compression (IVF-PQ)

## Experimental Results

Results are stored in `results/ALL_EXPERIMENTS.csv` with comprehensive metrics:

### Evaluation Metrics
- **Precision@k**: Precision at rank k (k=1,3,5,10,20,50)
- **Recall@k**: Recall at rank k
- **F1@k**: Harmonic mean of precision and recall
- **Hit@k**: Whether relevant document is in top-k results
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MRR**: Mean Reciprocal Rank
- **MAP**: Mean Average Precision
- **ANN Recall@k**: Approximate Nearest Neighbor recall

### Performance Metrics
- **Build time**: Index construction time (seconds)
- **Train time**: Training time (seconds)
- **Add time**: Time to add documents to index (seconds)
- **Search latency**: Query search time (mean, p50, p95, p99 in milliseconds)

### Key Findings

Sample results from experiments:

| Model | Index Method | Precision@10 | Recall@10 | NDCG@10 | Search (ms) |
|-------|--------------|--------------|-----------|---------|-------------|
| BGE   | HNSW         | 0.245        | 0.132     | 0.096   | 1.33        |
| BGE   | IVF-Flat     | 0.205        | 0.108     | 0.082   | 1.23        |
| E5    | HNSW         | 0.220        | 0.128     | 0.086   | 0.308       |
| E5    | IVF-Flat     | 0.198        | 0.108     | 0.078   | 2.30        |

## Running the Baseline

The main experiments are documented in `baseline2-qasper-dense-retrieval.ipynb`:

1. **Data Preparation**: Load and prepare QASPER dataset
2. **Embedding Generation**: Embed corpus documents using E5 or BGE
3. **Index Building**: Create dense retrieval indexes with various configurations
4. **Query Processing**: Embed queries and retrieve relevant documents
5. **Evaluation**: Compute comprehensive retrieval metrics
6. **Analysis**: Visualize results and compare methods

### Requirements

- Python 3.8+
- PyTorch with CUDA support (optional but recommended)
- transformers (for embedding models)
- faiss-gpu or faiss-cpu (for dense indexing)
- datasets (for QASPER download)
- pandas, numpy, matplotlib (for analysis)

## Future Work: Reinforcement Learning

The next phase of this project will:
- Train RL agents to learn optimal context selection policies
- Optimize for multiple objectives: accuracy, efficiency, and token budget
- Compare RL-based selection with baseline retrieval methods
- Evaluate on downstream QA tasks using selected context

