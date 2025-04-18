# utils_task_2/retrieval.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

from utils_task_2.query_decomposer import query_decomposer
from utils_task_2.embedding import (
    top_k_sections_by_similarity,
    embed_data_item_query,
    get_embeddings_parallel
)
from utils_task_2.constants import SECTION_ID_TO_NAME

def get_query_targets(user_query: str, k: int = 3) -> Dict:
    """
    Combine LLM‑inferred section plus top‑k by embedding similarity.
    """
    dec = query_decomposer(user_query)
    base = [dec["section_name"]]  # already internal ID
    sims = top_k_sections_by_similarity(dec["data_item"], k)
    # merge without dupes
    sections: List[str] = []
    for sid in base + sims:
        if sid not in sections:
            sections.append(sid)
    return {
        "ticker":      dec["ticker"],
        "year":        dec["year"],
        "section_ids": sections,
        "data_item":   dec["data_item"]
    }

def get_top_k_chunks(
    chunk_df,
    user_query: str,
    top_k_chunk: int = 3,
    top_k_section: int = 3,
    embedding_model: str = "text-embedding-3-small",
    include_neighbors: bool = False
) -> List[Dict]:
    """
    1) Decompose query → ticker, year, section_ids, data_item.
    2) Filter chunk_df by ticker & year.
    3) Embed data_item and each chunk_summary.
    4) For each section_id:
         - Compute similarity of query vs that section’s summaries.
         - Pick top_k_chunk summaries.
         - If `include_neighbors`, also grab the immediate prev/next chunk.
    5) Return a list of dicts with:
         ticker, year, section_id, section_name,
         chunk_summary, chunk, similarity.
    """
    # 1) Decompose
    targets     = get_query_targets(user_query, k=top_k_section)
    ticker      = targets["ticker"]
    year        = targets["year"]
    section_ids = targets["section_ids"]
    data_item   = targets["data_item"]

    # 2) Filter by ticker & year
    df_filt = chunk_df[
        (chunk_df.ticker == ticker) &
        (chunk_df.year   == year)
    ]

    # 3) Embed the query phrase
    q_emb = embed_data_item_query(data_item, model=embedding_model).reshape(1, -1)

    contexts = []
    seen = set()  # to dedupe (section_id, original_row_index)

    # 4) For each section pick top chunks (and optionally neighbors)
    for section_id in section_ids:
        sec_df = df_filt[df_filt.section == section_id].reset_index(drop=False)
        if sec_df.empty:
            continue

        summaries = sec_df["chunk_summary"].tolist()
        sec_embs  = np.vstack(get_embeddings_parallel(summaries, model=embedding_model))
        sims      = cosine_similarity(q_emb, sec_embs)[0]
        top_idxs  = sims.argsort()[::-1][:top_k_chunk]

        def collect(i: int):
            orig_idx = sec_df.at[i, "index"]
            key = (section_id, orig_idx)
            if key in seen:
                return
            seen.add(key)
            row = sec_df.iloc[i]
            contexts.append({
                "ticker":        row["ticker"],
                "year":          row["year"],
                "section_id":    section_id,
                "section_name":  SECTION_ID_TO_NAME[section_id],
                "chunk_summary": row["chunk_summary"],
                "chunk":         row["chunk"],
                "similarity":    float(sims[i])
            })

        for i in top_idxs:
            collect(i)
            if include_neighbors:
                if i - 1 >= 0:       collect(i - 1)
                if i + 1 < len(sec_df): collect(i + 1)

    # 5) sort by highest similarity
    contexts.sort(key=lambda x: x["similarity"], reverse=True)
    return contexts
