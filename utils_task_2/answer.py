# utils_task_2/answer.py

import json
from openai import OpenAI
from utils_task_2.retrieval import get_top_k_chunks
from utils_task_2.summarization import retry_on_exception
from utils_task_2.retrieval import get_query_targets

from llama_index.llms import openai
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    Document,
    ServiceContext,
    StorageContext,
    KnowledgeGraphIndex,
    Settings
)

client = OpenAI()

@retry_on_exception
def answer_query(
    chunk_df,
    user_query: str,
    top_k_chunk: int = 2,
    top_k_section: int = 2,
    include_neighbors: bool = True
) -> dict:
    """
    Retrieves contexts via get_top_k_chunks and then asks the LLM for:
      - answer     : string
      - explanation: which context numbers were used
      - relevance  : boolean
    """
    # 1) fetch contexts (with neighbors if desired)
    contexts = get_top_k_chunks(
        chunk_df,
        user_query,
        top_k_chunk=top_k_chunk,
        top_k_section=top_k_section,
        include_neighbors=include_neighbors
    )[:5]
    print(f"[INFO] Retrieved {len(contexts)} contexts for answering")

    # 2) build numbered summary block
    context_block = "\n".join(
        f"{i+1}. [{c['section_name']}] ({c['ticker']}, {c['year']}) "
        f"(sim={c['similarity']:.3f}): {c['chunk_summary']}"
        for i, c in enumerate(contexts)
    )

    system_prompt = f"""
You are an expert financial QA assistant. You have the following {len(contexts)} context chunks
extracted from SEC 10‑K filings, each labeled with its section, company, year, similarity score,
and a brief summary:

{context_block}

Your task:
- Answer the user's question **using only** the information in these contexts.
- Provide a concise **answer**.
- Provide a brief **explanation** stating which context numbers you used and why.
- Set **relevance** to true if those contexts fully support your answer, else false.

Respond **only** with JSON in this format:
{{"answer":"...", "explanation":"...", "relevance":true}}

Example:
{{"answer":"$1.2 billion","explanation":"Based on context 1 which reports net cash flow...","relevance":true}}
"""

    response = client.responses.create(
        model="gpt-4.1-2025-04-14",
        input=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_query}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "qa_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer":      {"type": "string"},
                        "explanation": {"type": "string"},
                        "relevance":   {"type": "boolean"}
                    },
                    "required": ["answer","explanation","relevance"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )

    return json.loads(response.output_text), contexts



def graphRAG_query(chunk_df, user_query, top_k_section=3):
    
    targets     = get_query_targets(user_query, k=top_k_section)
    ticker      = targets["ticker"]
    year        = targets["year"]
    section_ids = targets["section_ids"]
    data_item   = targets["data_item"]
    df_filt = chunk_df[
        (chunk_df.ticker == ticker) &
        (chunk_df.year   == year) &
        (chunk_df.section.isin(section_ids))
    ]

    # --- 1) Prepare your documents ---
    documents = [
        Document(text=row["chunk"])
        for _, row in df_filt.iterrows()
    ]

    # --- 2) Choose an LLM for *graph construction* (triplet extraction) ---
    Settings.llm = openai.OpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)

    # (Optionally also pick an embedder if you need to override default)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small"
    )

    # --- 3) Build your in‑memory graph index ---
    graph_store    = SimpleGraphStore()
    storage_ctx    = StorageContext.from_defaults(graph_store=graph_store)

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=10,
        storage_context=storage_ctx,
        include_embeddings=True
    )

    # --- 4) Now switch to your *query* LLM ---
    Settings.llm = openai.OpenAI(
        model="gpt-4.1-2025-04-14",  # higher‑capable model for final answers
        temperature=0.0,
    )
    
    query_engine = kg_index.as_query_engine(graph_traversal_depth=3)

    response = query_engine.query(
        user_query
    )
    
    return response.response, [node.text for node in response.source_nodes]
