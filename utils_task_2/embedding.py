# utils_task_2/embedding.py

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()

def get_embedding_single(text: str, model="text-embedding-3-small") -> list[float]:
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding

def get_embeddings_parallel(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    resp = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in resp.data]

def make_query_sentence(data_item: str) -> str:
    return f"This query is about {data_item.lower()} in the annual report."

def embed_data_item_query(data_item: str, model="text-embedding-3-small") -> np.ndarray:
    sent = make_query_sentence(data_item)
    return np.array(get_embedding_single(sent, model))

def top_k_sections_by_similarity(
    data_item: str, k=3, model="text-embedding-3-small"
) -> list[str]:
    from utils_task_2.constants import SECTION_DEFINITIONS, SECTION_NAME_TO_ID
    from sklearn.metrics.pairwise import cosine_similarity

    # prepare definitions & embeddings
    names = list(SECTION_DEFINITIONS.keys())
    texts = list(SECTION_DEFINITIONS.values())
    embs  = np.vstack(get_embeddings_parallel(texts, model))
    q_emb = embed_data_item_query(data_item, model).reshape(1,-1)

    sims   = cosine_similarity(q_emb, embs)[0]
    idxs   = sims.argsort()[::-1][:k]
    return [ SECTION_NAME_TO_ID[names[i]] for i in idxs ]
