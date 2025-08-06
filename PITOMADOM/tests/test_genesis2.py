"""Tests for genesis2 utilities."""

# flake8: noqa

import types
import pytest


@pytest.fixture
def genesis2(add_project_root):
    from inference import model as genesis2_module
    yield genesis2_module

class DummyTorch:
    @staticmethod
    def tensor(data, device=None):
        return data

    @staticmethod
    def cosine_similarity(a, b, dim=0):
        # simple cosine for scalars
        val = (a * b) / (abs(a) * abs(b))
        class Result(float):
            def item(self):
                return float(self)
        return Result(val)

    @staticmethod
    def no_grad():
        class _NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _NoGrad()

class DummyTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.vocab = {}
        self.reverse = {}

    def encode(self, text):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                idx = len(self.vocab) + 1
                self.vocab[word] = idx
                self.reverse[idx] = word
            ids.append(self.vocab[word])
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(self.reverse.get(t, "") for t in tokens)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return self.encode(messages[-1]["content"])

class DummyModel:
    def embed_tokens(self, tokens):
        # convert list of ints to floats
        return DummyEmbedding([float(t) for t in tokens])

class DummyEmbedding(list):
    def mean(self, dim=0):
        return sum(self) / len(self)

def test_genesis2_resonance_loop(monkeypatch, genesis2):
    tokenizer = DummyTokenizer()
    model = DummyModel()
    monkeypatch.setattr(genesis2, "torch", DummyTorch)

    def dummy_generate_fn(model, prompts, max_new_tokens, eos_token_id, temperature):
        return [tokenizer.encode("echo")]

    result = genesis2.genesis2_resonance_loop(
        model,
        tokenizer,
        "hello",
        dummy_generate_fn,
        iterations=5,
        resonance_threshold=0.5,
        device="cpu",
    )
    assert result.layers == 2
    assert result.final_resonance == "echo"
    assert result.evolution == 1.0

def test_random_delay(monkeypatch, genesis2):
    called = []
    monkeypatch.setattr(genesis2.random, "randint", lambda a, b: 1)
    monkeypatch.setattr(genesis2, "time", types.SimpleNamespace(sleep=lambda x: called.append(x)))
    genesis2.random_delay(min_seconds=1, max_seconds=5)
    assert called == [1]


def test_shutdown_executor(monkeypatch, genesis2):
    calls = []

    class DummyExecutor:
        def shutdown(self, wait=True):
            calls.append(wait)

    monkeypatch.setattr(genesis2, "_executor", DummyExecutor())
    genesis2.shutdown_executor()
    assert calls == [True]


def test_schedule_follow_up_invokes_callback(monkeypatch, genesis2):
    messages = []
    sleep_calls = []

    monkeypatch.setattr(genesis2.random, "random", lambda: 0.0)
    monkeypatch.setattr(genesis2.random, "randint", lambda a, b: 1)
    monkeypatch.setattr(
        genesis2, "time", types.SimpleNamespace(sleep=lambda s: sleep_calls.append(s))
    )
    monkeypatch.setattr(genesis2, "_follow_up_tasks", set())

    class DummyFuture:
        def cancel(self):
            return True

        def __hash__(self):
            return id(self)

    class SyncExecutor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)
            return DummyFuture()

    monkeypatch.setattr(genesis2, "_executor", SyncExecutor())

    history = [{"content": "test"}]
    task = genesis2.schedule_follow_up(history, lambda msg: messages.append(msg), probability=0.5)

    assert messages == [
        "I thought again about our discussion: 'test'. Here is an additional thought."
    ]
    assert sleep_calls == [1]
    assert task is not None


def test_schedule_follow_up_skips_when_probability_not_met(monkeypatch, genesis2):
    calls = []

    monkeypatch.setattr(genesis2.random, "random", lambda: 1.0)

    def fake_randint(a, b):
        calls.append((a, b))
        return 1

    monkeypatch.setattr(genesis2.random, "randint", fake_randint)

    class FailExecutor:
        def submit(self, fn, *args, **kwargs):
            raise AssertionError("submit should not be called")

    monkeypatch.setattr(genesis2, "_executor", FailExecutor())
    monkeypatch.setattr(genesis2, "_follow_up_tasks", set())

    history = [{"content": "test"}]
    messages = []
    task = genesis2.schedule_follow_up(history, lambda msg: messages.append(msg), probability=0.5)

    assert messages == []
    assert calls == []
    assert task is None


def test_cancel_follow_up_cancels_task(monkeypatch, genesis2):
    messages = []

    monkeypatch.setattr(genesis2.random, "random", lambda: 0.0)
    monkeypatch.setattr(genesis2.random, "randint", lambda a, b: 1)
    monkeypatch.setattr(genesis2, "time", types.SimpleNamespace(sleep=lambda s: None))
    monkeypatch.setattr(genesis2, "_follow_up_tasks", set())

    class DummyFuture:
        def __init__(self, fn):
            self.fn = fn
            self.cancelled = False

        def cancel(self):
            self.cancelled = True
            return True

        def run(self):
            if not self.cancelled:
                self.fn()

    class DummyExecutor:
        def submit(self, fn, *args, **kwargs):
            return DummyFuture(lambda: fn(*args, **kwargs))

    monkeypatch.setattr(genesis2, "_executor", DummyExecutor())

    history = [{"content": "test"}]
    task = genesis2.schedule_follow_up(history, lambda msg: messages.append(msg), probability=0.5)
    genesis2.cancel_follow_up(task)
    assert genesis2._follow_up_tasks == set()

    task.run()
    assert messages == []
