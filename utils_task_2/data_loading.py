# utils_task_2/data_loading.py

import io
import pandas as pd
import requests
from datasets import load_dataset, concatenate_datasets
from utils_task_2.constants import ALLOWED_YEARS, ALLOWED_TICKERS, SEC_HEADERS

def load_edgar_corpus(years=ALLOWED_YEARS):
    """
    Load the EDGAR corpus for the given years (train+validation+test splits)
    and concatenate into a single Hugging Face Dataset.
    """
    ds_list = [
        load_dataset("eloukas/edgar-corpus", f"year_{yr}", split="train+validation+test")
        for yr in years
    ]
    return concatenate_datasets(ds_list)

def load_cik_ticker_mapping():
    """
    Download the SEC’s ticker→CIK mapping and return as a DataFrame.
    """
    url = "https://www.sec.gov/include/ticker.txt"
    resp = requests.get(url, headers=SEC_HEADERS)
    resp.raise_for_status()
    df = pd.read_csv(
        io.StringIO(resp.text),
        sep=r"\s+",
        header=None,
        names=["ticker","cik"],
        dtype=str
    )
    return df

def filter_dataset_by_tickers(dataset, tickers=ALLOWED_TICKERS):
    """
    Given a HF Dataset, keep only filings whose CIK matches one of the provided tickers.
    """
    mapping = load_cik_ticker_mapping()
    mapping = mapping[mapping.ticker.isin(tickers)]
    valid_ciks = mapping.cik.unique().tolist()
    return dataset.filter(lambda row: row["cik"] in valid_ciks), mapping
