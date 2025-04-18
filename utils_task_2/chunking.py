# utils_task_2/chunking.py

import re
from utils_task_2.parsing import parse_paragraphs, group_paragraphs

def split_long_chunk_no_overlap(chunk: str, max_words: int = 500) -> list[str]:
    """
    Recursively split a long chunk by word count, preferring newline or sentence boundaries.
    """
    words = chunk.split()
    if len(words) <= max_words:
        return [chunk]

    left_approx = " ".join(words[:max_words])
    cut = len(left_approx)

    # try newline
    nl = chunk.rfind('\n', 0, cut)
    idx = nl if nl>0 else cut
    if nl<=0:
        # fallback to punctuation
        while idx>0 and chunk[idx-1] not in {'.','!','?'}:
            idx -= 1
        if idx==0:
            idx = cut

    left, right = chunk[:idx].strip(), chunk[idx:].strip()
    return [left] + split_long_chunk_no_overlap(right, max_words)

def split_chunks_with_overlap(chunks: list[str], max_words: int = 500) -> list[str]:
    """
    For a list of text chunks, split each to <=max_words but overlap
    the last sentence of the previous chunk into the next.
    """
    result, prev_last = [], None
    for chunk in chunks:
        subs = split_long_chunk_no_overlap(chunk, max_words)
        if prev_last:
            subs[0] = f"{prev_last} {subs[0]}"
        result.extend(subs)
        # capture last sentence
        sents = re.split(r'(?<=[\.!\?])\s+', subs[-1])
        prev_last = sents[-1] if sents else None
    return result

def split_into_chunks(text: str, min_word_threshold: int = 10) -> list[str]:
    """
    1) parse raw text into cleaned paragraphs
    2) group them into logical chunks
    Returns a list of text chunks.
    """
    final_chunks = []
    paras = parse_paragraphs(text, min_word_threshold)
    chunks = group_paragraphs(paras, min_word_threshold)
    for chunk in chunks:
        final_chunks.extend(split_long_chunk_no_overlap(chunk, max_words=300))
    return final_chunks
