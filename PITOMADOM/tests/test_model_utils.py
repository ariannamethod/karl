import pytest

"""Tests for monolithic model utilities."""

# flake8: noqa


@pytest.fixture
def model_utils(add_project_root):
    from inference.model import append_message, ByteTokenizer
    return append_message, ByteTokenizer


def test_message_history_limit(model_utils):
    append_message, _ = model_utils
    messages = [{"role": "system", "content": "sys"}]
    for i in range(5):
        append_message(messages, {"role": "user", "content": f"u{i}"}, 2)
        append_message(messages, {"role": "assistant", "content": f"a{i}"}, 2)
    assert len(messages) == 1 + 2 * 2
    assert messages[1]["content"] == "u3"
    assert messages[2]["content"] == "a3"
    assert messages[3]["content"] == "u4"
    assert messages[4]["content"] == "a4"


def test_byte_tokenizer_roundtrip(model_utils):
    _, ByteTokenizer = model_utils
    tok = ByteTokenizer()
    text = "hello"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
