# 10‑K Filing Analysis & QA Pipeline

## Project Overview  
This project demonstrates a three‑stage pipeline for analyzing SEC 10‑K filings:  
1. **Task 1 – Engineering**  
   - Chunk filings into coherent segments, embed them, reduce to 2D, cluster and flag outliers.  
2. **Task 2 – Gen AI Extraction**  
   - Build an embedding‑based retrieval + LLM extraction system to answer factual queries.  
3. **Task 3 – Chatbot (GraphRAG)**  
   - Wrap the same QA pipeline in a graph‑augmented retrieval agent for richer, context‑aware answers.

---

## Prerequisites  
- Python 3.8+  
- An **OpenAI API key** with quota for embeddings and completions (set it as `OPENAI_API_KEY`).

---

## Installation

1. Clone this repo and `cd` into its root directory.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the entire pipeline end‑to‑end with:
```bash
python main.py
```

> **Warning:**  
> - **Task 1** (embeddings + PCA + clustering) runs on CPU and can take **~10 minutes**.  
> - **Tasks 2 & 3** (retrieval + LLM prompting & GraphRAG) may take **significantly longer**, especially the GraphRAG stage, due to its graph‑traversal overhead.

---

## Configuration

- Make sure your OpenAI API key is available to the script:
  ```bash
  export OPENAI_API_KEY="your_api_key_here"
  ```
- You can adjust key parameters (chunk size, PCA components, number of clusters, graph depth) in `config.py`.

---

## Project Structure

```
.
├── main.py               # Entry point: orchestrates Task 1, 2 & 3
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── config.py             # Hyperparameters & API settings
├── utils/
│   ├── parsing.py        # Text → paragraph/chunk parsing logic
│   ├── chunking.py       # Heuristic chunk grouping
│   ├── embeddings.py     # Sentence‑BERT embedding wrappers
│   ├── pca.py            # PCA & variance analysis
│   ├── clustering.py     # KMeans + silhouette scoring
│   └── graph_rag.py      # GraphRAG construction & query interface
├── data/
│   └── raw/              # Place your 10‑K text files here
└── outputs/
    ├── plots/            # PCA, silhouette, t‑SNE visualizations
    └── results/          # QA answers & logs
```

---

## What’s Next?

- **Optimize** graph‑based queries (caching, pruning) to reduce Task 3 latency.  
- **Expand** to more companies, filing years, and query types.  
- **Automate** hyperparameter tuning based on dataset and query complexity.

Enjoy exploring 10‑K filings with embeddings and generative AI!  

