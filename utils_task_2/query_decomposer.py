# utils_task_2/query_decomposer.py

import json, logging
from openai import OpenAI
from utils_task_2.constants import ALLOWED_TICKERS, ALLOWED_YEARS, SECTION_NAME_TO_ID
from utils_task_2.summarization import retry_on_exception
from utils_task_2.logging_utils import log_usage

client = OpenAI()
ALLOWED_SECTIONS = list(SECTION_NAME_TO_ID.keys())

@retry_on_exception
def query_decomposer(user_query: str) -> dict:
    """
    Ask an LLM to extract:
      - ticker    (one of ALLOWED_TICKERS)
      - year      (one of ALLOWED_YEARS)
      - section_name (one of ALLOWED_SECTIONS)
      - data_item   (the exact phrase user wants)
    Returns all four, with section_name mapped to internal ID.
    """
    system = f"""
You are an expert in SEC 10‑K analysis. Extract four fields:
1) ticker – one of {ALLOWED_TICKERS}
2) year   – one of {ALLOWED_YEARS}
3) section_name – choose from {ALLOWED_SECTIONS}. If user only mentions a metric,
                   infer the best matching section title.
4) data_item – the exact phrase or term the user used.

Respond only with JSON like:
{{"ticker":"MSFT","year":"2019","section_name":"Financial Statements","data_item":"net cash flow"}}
"""
    resp = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        input=[
            {"role":"system","content":system},
            {"role":"user",  "content":user_query}
        ],
        text={
            "format":{
                "type":"json_schema",
                "name":"query_decomposition",
                "schema":{
                    "type":"object",
                    "properties":{
                        "ticker":{"type":"string","enum":ALLOWED_TICKERS},
                        "year":  {"type":"string","enum":ALLOWED_YEARS},
                        "section_name":{"type":"string","enum":ALLOWED_SECTIONS},
                        "data_item":{"type":"string"}
                    },
                    "required":["ticker","year","section_name","data_item"],
                    "additionalProperties":False
                },
                "strict":True
            }
        }
    )

    log_usage(resp.usage, "query_decomposer")
    out = json.loads(resp.output_text)
    # map to internal ID
    out["section_name"] = SECTION_NAME_TO_ID[out["section_name"]]
    return out
