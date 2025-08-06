import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.async_generate import get_parser, process_dataset  # noqa: E402


def test_parser_parses_args() -> None:
    parser = get_parser()
    args = parser.parse_args([
        "--dataset",
        "dummy",
        "--output",
        "out.jsonl",
        "--num-completions",
        "2",
        "--concurrency",
        "3",
    ])
    assert args.dataset == "dummy"
    assert args.output == "out.jsonl"
    assert args.num_completions == 2
    assert args.concurrency == 3


def test_process_dataset_schedules(monkeypatch, tmp_path: Path) -> None:
    class DummyDataset:
        def __init__(self) -> None:
            self.data = [{"prompt": "hello"}, {"prompt": "world"}]

        def __len__(self) -> int:
            return len(self.data)

        def __iter__(self):
            return iter(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    dataset = DummyDataset()

    monkeypatch.setattr("scripts.async_generate.load_dataset", lambda *_, **__: dataset)

    calls = {"count": 0}

    async def fake_fetch(*args, **kwargs):  # type: ignore[unused-ignore]
        calls["count"] += 1
        return "ok"

    monkeypatch.setattr("scripts.async_generate.fetch_completion", fake_fetch)

    out_file = tmp_path / "out.jsonl"
    parser = get_parser()
    args = parser.parse_args([
        "--dataset",
        "dummy",
        "--output",
        str(out_file),
        "--num-completions",
        "2",
    ])

    asyncio.run(process_dataset(args))

    with out_file.open() as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert calls["count"] == 4
