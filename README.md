# 10‑K Filing Analysis & QA Pipeline

## Project Overview  
This project implements a three‑stage pipeline for analyzing SEC 10‑K filings:  
1. **Task 1 – Engineering**  
   - Chunk filings into coherent segments, embed them, reduce to 2D, cluster and flag outliers.  
2. **Task 2 – Gen AI Extraction**  
   - Build an embedding‑based retrieval + LLM extraction system to answer factual queries.  
3. **Task 3 – Chatbot (GraphRAG)**  
   - Wrap the same QA pipeline in a graph‑augmented retrieval agent for richer, context‑aware answers.

Alongside the code, you’ll find **Reports.pdf**, which contains my detailed summary and reflections on the project.

---

## Prerequisites  
- Python 3.8+  
- Your own OpenAI API key (no environment export needed—`main.py` will prompt you to enter it manually).

---

## Installation

1. Clone this repo and `cd` into its root directory.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the pipeline end‑to‑end with:
```bash
python main.py
```
When you run **main.py**, you’ll be prompted to enter your OpenAI API key in the CLI.

> **Warning:**  
> - **Task 1** (CPU‑based embeddings + PCA + clustering) can take **around 10 minutes**.  
> - **Tasks 2 & 3** (LLM retrieval & GraphRAG) may take **significantly longer**, due to graph traversal overhead.

---

## Project Structure

```
.
├── main.py               # Entry point: prompts for API key, orchestrates all tasks
├── requirements.txt      # Python dependencies
├── Reports.pdf           # Final report and personal reflections
├── README.md             # This document
├── config.py             # Hyperparameters & pipeline settings
├── utils_task_1/         # Utility functions and scripts for Task 1 (parsing, chunking, PCA, clustering)
├── utils_task_2/         # Utility functions and scripts for Task 2 & 3 (retrieval pipelines, graph RAG)
└── plots/                # Plots generated by Task 1 (PCA variance, silhouette scores, t‑SNE maps)
```

---

## What’s Next?

- **Optimize** GraphRAG speed (caching, pruning, parallelism) to shorten Task 3 runtime.  
- **Expand** to more companies, filing years, and more diverse query types.  
- **Automate** hyperparameter tuning based on dataset size and query complexity.

Enjoy exploring 10‑K filings with embeddings and generative AI!

