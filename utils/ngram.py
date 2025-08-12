from __future__ import annotations

from collections import Counter
from typing import Callable, List, Tuple
import math

TokenList = List[str]
Tokenizer = Callable[[str], TokenList]


def default_tokenizer(text: str) -> TokenList:
    """Split ``text`` into tokens using whitespace."""
    return text.split()


class NGramModel:
    """Simple n-gram language model with optional smoothing.

    Parameters
    ----------
    n: int
        Order of the model (size of the n-grams).
    tokenizer: Callable[[str], List[str]], optional
        Function to split input text into tokens. Defaults to ``str.split``.
    smoothing: str, optional
        Type of smoothing to apply. Supported values: ``'laplace'`` and
        ``'kneser-ney'``. If ``None`` or unrecognised, no smoothing is used.
    discount: float, optional
        Discount parameter for Kneser–Ney smoothing.
    """

    def __init__(
        self,
        n: int = 2,
        tokenizer: Tokenizer | None = None,
        smoothing: str | None = None,
        discount: float = 0.75,
    ) -> None:
        self.n = max(1, n)
        self.tokenizer = tokenizer or default_tokenizer
        self.smoothing = (smoothing or "").lower()
        self.discount = discount
        self.ngram_counts: Counter[Tuple[str, ...]] = Counter()
        self.context_counts: Counter[Tuple[str, ...]] = Counter()
        self.vocab: set[str] = set()

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------
    def fit(self, text: str) -> None:
        """Update the model with ``text``."""
        tokens = self.tokenizer(text)
        self.vocab.update(tokens)
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            context = ngram[:-1]
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    # ------------------------------------------------------------------
    # Probability and perplexity
    # ------------------------------------------------------------------
    def _laplace_prob(self, ngram: Tuple[str, ...], context_count: int) -> float:
        vocab_size = len(self.vocab)
        count = self.ngram_counts.get(ngram, 0)
        return (count + 1) / (context_count + vocab_size)

    def _kneser_ney_prob(self, ngram: Tuple[str, ...], context_count: int) -> float:
        if self.n != 2:
            raise ValueError("Kneser–Ney smoothing implemented for bigrams only")
        if context_count == 0:
            return 0.0
        discount = self.discount
        count = self.ngram_counts.get(ngram, 0)
        # Number of unique continuations for the context
        cont_context = len({w2 for (w1, w2) in self.ngram_counts if w1 == ngram[0]})
        # Number of contexts that word has appeared in
        cont_word = len({w1 for (w1, w2) in self.ngram_counts if w2 == ngram[1]})
        total_contexts = len(self.context_counts)
        return max(count - discount, 0) / context_count + (
            discount * cont_context / context_count * (cont_word / total_contexts)
        )

    def probability(self, ngram: Tuple[str, ...]) -> float:
        context = ngram[:-1]
        context_count = self.context_counts.get(context, 0)
        if self.smoothing == "laplace":
            return self._laplace_prob(ngram, context_count)
        elif self.smoothing in {"kneser-ney", "kneser", "kneserney"}:
            return self._kneser_ney_prob(ngram, context_count)
        if context_count == 0:
            return 0.0
        return self.ngram_counts.get(ngram, 0) / context_count

    def perplexity(self, text: str) -> float:
        tokens = self.tokenizer(text)
        if len(tokens) < self.n:
            return float("inf")
        log_sum = 0.0
        count = 0
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            prob = self.probability(ngram)
            if prob == 0:
                return float("inf")
            log_sum += -math.log(prob)
            count += 1
        return math.exp(log_sum / max(count, 1))
