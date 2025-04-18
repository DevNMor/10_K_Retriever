"""
Parsing utilities: convert raw text to cleaned paragraphs and group them.
"""

def parse_paragraphs(text, min_word_threshold=10, sentence_endings={'.','!','?'}):
    """
    Splits raw text into lines, filters empties, then merges lines lacking terminal punctuation.
    Returns a list of cleaned paragraphs.
    """
    raw = [line.strip() for line in text.splitlines() if line.strip()]
    merged = []
    buffer = None
    for line in raw:
        if buffer is None:
            buffer = line
        else:
            if buffer[-1] not in sentence_endings:
                buffer += ' ' + line
            else:
                merged.append(buffer)
                buffer = line
    if buffer:
        merged.append(buffer)
    return merged


def is_strange(para, min_word_threshold=10, ending_punctuations={'.','!','?'}):
    """
    Flags paragraphs that:
      - lack ending punctuation,
      - have fewer than min_word_threshold words,
      - or are title-cased / ALL-CAPS (likely headers).
    """
    text = para.strip()
    if not text:
        return True
    if text[-1] not in ending_punctuations:
        return True
    words = text.split()
    if len(words) < min_word_threshold:
        return True
    if text.istitle() or text.isupper():
        return True
    return False


def group_paragraphs(paragraphs, min_word_threshold=10):
    """
    Groups paragraphs starting at a 'strange' (header) paragraph,
    collecting until the next 'strange' after at least one normal paragraph.
    Returns a list of text chunks.
    """
    groups = []
    current = []
    has_normal = False
    for para in paragraphs:
        flag = is_strange(para, min_word_threshold)
        if not current:
            if flag:
                current = [para]
                has_normal = False
            else:
                groups.append(para)
        else:
            if flag and has_normal:
                groups.append("\n".join(current))
                current = [para]
                has_normal = False
            else:
                current.append(para)
                if not flag:
                    has_normal = True
    if current:
        groups.append("\n".join(current))
    return groups