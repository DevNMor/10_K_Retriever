# main.py
print("Initializing ...")
import argparse
import json
import logging
import os

os.environ["OPENAI_API_KEY"] = input("Please input your OpenAI API Key to proceed:\n")

from datasets import load_dataset
from sklearn.manifold import TSNE

import pandas as pd

# Task 1 modules (fully qualified imports to avoid name clashes)
import utils_task_1.parsing       as ut1_parsing
import utils_task_1.chunking      as ut1_chunking
import utils_task_1.embedding     as ut1_embedding
import utils_task_1.pca           as ut1_pca
import utils_task_1.clustering    as ut1_clustering
import utils_task_1.visualization as ut1_vis

# Task 2 modules
import utils_task_2.data_loading     as ut2_data
import utils_task_2.chunking         as ut2_chunking
import utils_task_2.summarization    as ut2_summarization
import utils_task_2.query_decomposer as ut2_decomposer
import utils_task_2.embedding        as ut2_embedding
import utils_task_2.retrieval        as ut2_retrieval
import utils_task_2.answer           as ut2_answer

from utils_task_2.constants import TEST_QUERIES


def setup_logging():
    # 1) Set your root level (or whatever you like)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # 2) Silence OpenAI client info logs
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)

    # 3) Silence the HTTP stack (httpx / httpcore / anyio)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anyio").setLevel(logging.WARNING)

def task_1_pipeline():
    """Runs the Task 1 clustering & visualization pipeline."""
    logging.info("=== Task 1 Pipeline Started ===")

    # 1. Load and sample
    logging.info("Loading EDGAR 10-K filings for 2020...")
    ds2020 = load_dataset("eloukas/edgar-corpus", "year_2020", split="train+validation+test")
    sample = ds2020.shuffle(seed=42)[:10]
    logging.info(f"Sampled {len(sample)} filings.")

    # 2. Load model & tokenizer
    logging.info("Loading model and tokenizer...")
    model, tokenizer = ut1_embedding.load_model_and_tokenizer()

    # 3. Parse, group, chunk
    logging.info("Parsing and grouping paragraphs into chunks...")
    chunks, labels = [], []
    for sec in sample:
        if not sec.startswith("section_"):
            continue
        for text in sample[sec]:
            paras  = ut1_parsing.parse_paragraphs(text)
            grouped = ut1_parsing.group_paragraphs(paras)
            for g in grouped:
                subs = ut1_chunking.split_long_chunk(g, tokenizer)
                chunks.extend(subs)
                labels.extend([sec] * len(subs))
    logging.info(f"Generated {len(chunks)} text chunks.")

    # 4. Embedding
    logging.info("Computing embeddings...")
    embs = ut1_embedding.compute_embeddings(model, chunks)

    # 5. Scale & PCA
    logging.info("Standard scaling embeddings...")
    embs_scaled = ut1_clustering.standard_scale_embeddings(embs)
    logging.info("Plotting PCA variance curve...")
    os.makedirs('plots', exist_ok=True)
    ut1_vis.plot_pca_variance(embs_scaled, save_path='plots/pca_variance.png')

    logging.info("Reducing with PCA to 200 dims...")
    pca_embs, pca_model = ut1_pca.compute_pca(embs_scaled, n_components=200)
    X_norm = ut1_pca.normalize_rows(pca_embs)

    # 6. Choose K & KMeans
    logging.info("Choosing K via silhouette analysis...")
    best_k, _ = ut1_clustering.choose_k_by_silhouette(X_norm, k_min=15, k_max=80)
    logging.info(f"Best K = {best_k}")
    labels_k, centroids = ut1_clustering.perform_kmeans(X_norm, best_k)

    # 7. Outlier detection
    logging.info("Detecting outliers...")
    outliers = ut1_clustering.detect_outliers(X_norm, labels_k, centroids, percentile=90)

    # 8. t-SNE & visualization
    logging.info("Computing t-SNE projections...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embs_2d = tsne.fit_transform(X_norm)

    logging.info("Plotting clusters...")
    ut1_vis.plot_clusters(embs_2d, labels_k, path='plots/clusters.png')
    logging.info("Plotting outliers...")
    ut1_vis.plot_outliers(embs_2d, outliers, path='plots/outliers.png')
    logging.info("Plotting section labels...")
    ut1_vis.plot_sections(embs_2d, labels, path='plots/sections.png',
                          section_names=sorted(set(labels)))

    logging.info("=== Task 1 Pipeline Completed ===")

def task_2_3_pipeline():
    """Runs the Task 2 and 3: RAG & GraphRAG QA pipeline for query testing."""

    logging.info("=== Task 2 & 3 Pipeline Started ===")

    # 1. Load and filter dataset
    logging.info("Loading and filtering EDGAR corpus for selected tickers...")
    ds = ut2_data.load_edgar_corpus()
    ds, sample_company = ut2_data.filter_dataset_by_tickers(ds)

    # 2. Build chunk_df
    logging.info("Chunking and summarizing filings...")
    rows = []
    for report in ds:
        for sec, text in report.items():
            if not sec.startswith("section_"):
                continue
            for chunk in ut2_chunking.split_into_chunks(text):
                rows.append({
                    "chunk":         chunk,
                    "cik":           report["cik"],
                    "ticker":        sample_company[sample_company.cik==report['cik']].ticker.item(),
                    "year":          report["year"],
                    "section":       sec
                })
    chunk_df = pd.DataFrame(rows)

    logging.info("Generating chunk summaries in parallel...")
    chunk_df = ut2_summarization.parallel_summarize(chunk_df)

    # 3. For each test, decompose, retrieve, answer, then print vs. ground truth
    for i, test in enumerate(TEST_QUERIES, 1):
        q = test["query"]
        gt = test["ground_truth"]
        logging.info(f"Test #{i}: query={q!r}")
        try:
            result, _ = ut2_answer.answer_query(
                chunk_df, q,
                top_k_chunk=3,
                top_k_section=3,
                include_neighbors=False
            )

            result_graph, _ = ut2_answer.graphRAG_query(
                chunk_df, q, top_k_section=3
            )
        except Exception as e:
            logging.error(f"Failed to answer test #{i}: {e}")
            continue

        print(f"\n=== Test #{i} ===")
        print(f"Query       : {q}")
        print(f"Ground truth: {gt}")
        print("RAG result  :")
        print(json.dumps(result, indent=2))
        print("Graph RAG result  :")
        print(json.dumps(result, indent=2))

    logging.info("=== Task 2 & 3 Test Harness Completed ===")

if __name__ == '__main__':
    setup_logging()
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "<YOUR_API_KEY_HERE>")
    setup_logging()
    task_1_pipeline()
    task_2_3_pipeline()
