# utils_task_2/summarization.py

import time, json, logging
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from utils_task_2.logging_utils import log_usage

client = OpenAI()

def retry_on_exception(fn):
    """Retry decorator: up to 3 tries with 1s backoff."""
    def wrapped(*args, **kwargs):
        last_exc = None
        for i in range(3):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                time.sleep(1)
        raise last_exc
    return wrapped

@retry_on_exception
def summarizer(chunk_text: str) -> str:
    """LLM call: return a one‑sentence JSON summary."""
    system = """You are a financial‑report summarizer. Read the excerpt and
produce one concise sentence (15–25 words) capturing its main point and significance.
Return only: {"response":"..."}"""
    resp = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        input=[
            {"role":"system","content":system},
            {"role":"user",  "content":chunk_text}
        ],
        text={
            "format":{
                "type":"json_schema",
                "name":"financial_report_excerpt_summarization",
                "schema":{
                    "type":"object",
                    "properties":{"response":{"type":"string"}},
                    "required":["response"],
                    "additionalProperties":False
                },
                "strict":True
            }
        }
    )
    # log_usage(resp.usage, "summarizer")
    return json.loads(resp.output_text)["response"]

def safe_summarizer(chunk_text: str):
    """Wrap summarizer, log on failure, return None."""
    try:
        return summarizer(chunk_text)
    except Exception as e:
        print(f"[Warning] summarization failed: {e}")
        return None

def parallel_summarize(df, text_column="chunk", summary_column="chunk_summary"):
    """
    Summarize df[text_column] in parallel, store into df[summary_column].
    """
    chunks = df[text_column].tolist()
    with ThreadPoolExecutor() as exe:
        results = list(tqdm(
            exe.map(safe_summarizer, chunks),
            total=len(chunks), desc="Summarizing"
        ))
    df[summary_column] = results
    return df
