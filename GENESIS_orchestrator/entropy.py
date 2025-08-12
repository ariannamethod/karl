from collections import Counter
from collections.abc import Iterable
import math
from typing import Generator


def ngram_counter(stream: Iterable[str], n: int) -> Generator[Counter[str], None, None]:
    """Yield cumulative n-gram counts for chunks from the stream.

    Parameters
    ----------
    stream: Iterable[str]
        Iterable producing pieces of text.
    n: int
        Order of the n-grams to count. Must be positive.

    Yields
    ------
    Counter[str]
        Cumulative counts of observed n-grams after processing each chunk.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    counts: Counter[str] = Counter()
    buffer = ""
    for chunk in stream:
        if not isinstance(chunk, str):
            raise TypeError("stream must yield strings")
        data = buffer + chunk
        if len(data) >= n:
            for i in range(len(data) - n + 1):
                counts[data[i:i + n]] += 1
            buffer = data[-(n - 1):] if n > 1 else ""
        else:
            buffer = data
        yield counts.copy()


def markov_entropy(text: str | Iterable[str], n: int = 2) -> float:
    """Compute Shannon entropy of character n-grams in ``text``.

    ``text`` may be a single string or an iterable producing strings.
    """
    if isinstance(text, str):
        stream: Iterable[str] = [text]
    elif isinstance(text, Iterable):
        stream = text
    else:
        raise TypeError("text must be a string or an iterable of strings")

    final_counts: Counter[str] | None = None
    for counts in ngram_counter(stream, n):
        final_counts = counts
    if not final_counts:
        return 0.0
    total = sum(final_counts.values())
    return -sum((c / total) * math.log2(c / total) for c in final_counts.values())
