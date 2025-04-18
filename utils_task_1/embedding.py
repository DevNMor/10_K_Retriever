"""
Model loading and embedding computation.
"""
from sentence_transformers import SentenceTransformer

def load_model_and_tokenizer(model_name='sentence-transformers/all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    return model, model.tokenizer


def compute_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)