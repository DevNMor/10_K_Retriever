"""
Split overlong chunks into sub-chunks at nearest sentence/newline.
"""

def split_long_chunk(chunk, tokenizer):
    """
    Ensures each sub-chunk token count ≤ model_max_length by:
    1. Tokenizing and checking length
    2. Cutting at the nearest '.', '!', '?', or '\n' before limit
    3. Recursing on remainder
    Requires a valid tokenizer instance.
    """
    # Use provided tokenizer only
    max_len = tokenizer.model_max_length
    ids = tokenizer.encode(chunk, add_special_tokens=True)
    if len(ids) <= max_len:
        return [chunk]
    # Determine approximate cut position in characters
    cut = min(len(chunk), len(tokenizer.decode(ids[:max_len], skip_special_tokens=True)))
    idx = cut
    # Scan backwards for natural boundary
    while idx > 0 and chunk[idx-1] not in ['\n', '.', '!', '?']:
        idx -= 1
    if idx == 0:
        idx = cut
    left = chunk[:idx].strip()
    right = chunk[idx:].strip()
    return [left] + split_long_chunk(right, tokenizer)