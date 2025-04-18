# utils_task_2/parsing.py

def parse_paragraphs(text, min_word_threshold=10, sentence_endings={'.','!','?'}):
    """
    1) Split raw text into non-empty lines.
    2) Merge lines that don’t end in punctuation into a single paragraph.
    Returns a list of cleaned paragraphs.
    """
    raw = [line.strip() for line in text.splitlines() if line.strip()]
    merged, buffer = [], None
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
    Heuristic: flag paragraphs with no ending punctuation,
    too few words, or header‑like casing.
    """
    text = para.strip()
    if not text or text[-1] not in ending_punctuations:
        return True
    words = text.split()
    if len(words) < min_word_threshold:
        return True
    if text.istitle() or text.isupper():
        return True
    return False

def group_paragraphs(paragraphs, min_word_threshold=10):
    """
    Group “strange” header paras together with following normal paras
    until the next header, producing logical text chunks.
    """
    groups, current, has_normal = [], [], False
    for para in paragraphs:
        flag = is_strange(para, min_word_threshold)
        if not current:
            if flag:
                current, has_normal = [para], False
            else:
                groups.append(para)
        else:
            if flag and has_normal:
                groups.append("\n".join(current))
                current, has_normal = [para], False
            else:
                current.append(para)
                if not flag:
                    has_normal = True
    if current:
        groups.append("\n".join(current))
    return groups
