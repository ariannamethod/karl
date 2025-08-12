import re

import pytest

from utils.ngram import NGramModel

def test_custom_tokenizer_by_words():
    text = "Hello, world!"
    model = NGramModel(n=2, tokenizer=lambda s: re.findall(r"\w+", s))
    model.fit(text)
    assert ("Hello", "world") in model.ngram_counts

def test_laplace_smoothing_nonzero_probability():
    model = NGramModel(n=2, smoothing="laplace")
    model.fit("hello world")
    assert model.probability(("world", "hello")) > 0

def test_kneser_ney_smoothing_nonzero_probability():
    model = NGramModel(n=2, smoothing="kneser-ney")
    # context "world" appears but "world hello" bigram does not
    model.fit("hello world hello")
    assert model.probability(("world", "hello")) > 0

def test_perplexity_consistent_across_lengths():
    long_text = "a b " * 50
    short_text = "a b a b"
    model = NGramModel(n=2, smoothing="laplace")
    model.fit(long_text)
    assert model.perplexity(short_text) == pytest.approx(
        model.perplexity(long_text), rel=1e-2
    )
