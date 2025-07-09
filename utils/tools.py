def split_message(text: str, max_length: int = 4000):
    parts = []
    while len(text) > max_length:
        cut = text.rfind("\n", 0, max_length)
        if cut < 0: cut = max_length
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts