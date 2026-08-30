# Reinforcement-Learning-for-Query-Focused-Context-Selection-from-Long-Meeting-Transcripts

We develop an adaptive retrieval system for long meeting transcripts that selects relevant evidence under a token budget. We compare BM25, dense retrieval, reranking, and sequential methods before using reinforcement learning to optimize evidence selection, accuracy, and context efficiency.

## Approach

This repository evaluates multiple baseline retrieval strategies on the QASPER dataset before introducing advanced adaptive methods:
- **Baseline 1 (BM25):** Sparse retrieval using BM25.
- **Baseline 2 (Dense Retrieval):** Dense retrieval using BGE embeddings and FAISS HNSW index.
- **Baseline 3 (Hybrid Retrieval):** Combines Baseline 1 and Baseline 2 using Reciprocal Rank Fusion (RRF, constant 60) and weighted fusion (sweeping $\alpha$ from 0.0 to 1.0 in steps of 0.1).

## Results

Evaluation of the hybrid retrieval (Baseline 3) shows that combining dense and sparse signals improves overall retrieval performance compared to using either independently:
- **BM25 Only (Sparse):** Achieves an MRR of ~0.085 and MAP of ~0.058.
- **Dense Only (BGE):** Achieves an MRR of ~0.100 and MAP of ~0.066.
- **Hybrid Fusion (RRF & Weighted):** The best weighted fusion approach ($\alpha \approx 0.5$) achieves an MRR of ~0.112 and MAP of ~0.076. Similarly, RRF yields an MRR of ~0.111 and MAP of ~0.075.

The hybrid method successfully captures complementary evidence from both lexical (BM25) and semantic (Dense) spaces, yielding higher recall and precision metrics across all top-K cutoffs.
