"""Data collection and orchestration utilities for GENESIS."""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .orchestrator import (
    STATE_FILE,
    dataset_dir as CONFIG_DATASET_DIR,
    file_hash,
    load_state,
    model_hyperparams,
    save_state,
    threshold as DEFAULT_THRESHOLD,
)
from .entropy import markov_entropy, model_perplexity
from .genesis_trainer import prepare_char_dataset, train_model

LOGGER = logging.getLogger(__name__)
DATASET_FILE = Path(__file__).with_name("gen_data.txt")

# Default set of file extensions considered for processing.
DEFAULT_ALLOW_EXT = {".txt", ".md", ".py"}


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(1024)
        return b"\0" in chunk
    except Exception:
        return True


def _iter_text_files(
    paths: Iterable[Path],
    exclude: Iterable[Path],
    allow_ext: Iterable[str] = DEFAULT_ALLOW_EXT,
    deny_ext: Optional[Iterable[str]] = None,
) -> Iterable[Path]:
    """Yield text files, filtered by allow/deny extension lists."""

    exclude = {p.resolve() for p in exclude}
    allow = {e.lower() for e in allow_ext} if allow_ext else None
    deny = {e.lower() for e in deny_ext} if deny_ext else set()
    for base in paths:
        for path in base.rglob("*"):
            if not path.is_file() or path.resolve() in exclude or _looks_binary(path):
                continue
            suffix = path.suffix.lower()
            if allow is not None and suffix not in allow:
                continue
            if suffix in deny:
                continue
            yield path


def collect_new_data(
    base_paths: Iterable[Path],
    dataset_path: Path = DATASET_FILE,
    threshold: int = DEFAULT_THRESHOLD,
    resume: bool = False,
    logger: Optional[logging.Logger] = None,
    allow_ext: Iterable[str] = DEFAULT_ALLOW_EXT,
    deny_ext: Optional[Iterable[str]] = None,
    chunk_size: int = 8192,
) -> Tuple[bool, str]:
    """Collect text from ``base_paths`` and write to ``dataset_path`` when threshold exceeded.

    Files are streamed line by line to avoid holding large intermediate strings in memory.
    ``chunk_size`` controls how many bytes are buffered before flushing to the final
    output string.
    """

    log = logger or LOGGER
    file_state = load_state() if resume else {}
    temp_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    total = 0
    data_parts: list[str] = []
    buffer: list[str] = []
    buffer_bytes = 0

    def flush_buffer() -> None:
        nonlocal buffer_bytes
        if buffer:
            data_parts.append("".join(buffer))
            buffer.clear()
            buffer_bytes = 0

    with temp_path.open("w", encoding="utf-8") as tmp:
        first = True
        nl = "\n"
        nl_bytes = len(nl.encode("utf-8"))
        for path in _iter_text_files(
            base_paths,
            exclude=[dataset_path, STATE_FILE, temp_path],
            allow_ext=allow_ext,
            deny_ext=deny_ext,
        ):
            try:
                size = path.stat().st_size
                h = file_hash(path, size=size)
                if h is None:
                    continue
            except Exception:
                log.exception("failed to hash %s", path)
                continue
            key = str(path.resolve())
            entry = file_state.get(key)
            if entry and entry.get("hash") == h and entry.get("size") == size:
                continue

            def process_file() -> bool:
                nonlocal total, buffer_bytes
                if not first:
                    tmp.write(nl)
                    buffer.append(nl)
                    total += nl_bytes
                    buffer_bytes += nl_bytes
                    if buffer_bytes >= chunk_size:
                        flush_buffer()
                try:
                    with path.open("r", encoding="utf-8") as src:
                        for line in src:
                            tmp.write(line)
                            buffer.append(line)
                            line_bytes = len(line.encode("utf-8"))
                            total += line_bytes
                            buffer_bytes += line_bytes
                            if buffer_bytes >= chunk_size:
                                flush_buffer()
                    return True
                except Exception:
                    log.exception("failed to read %s", path)
                    try:
                        with path.open("r", encoding="utf-8", errors="ignore") as src:
                            for line in src:
                                tmp.write(line)
                                buffer.append(line)
                                line_bytes = len(line.encode("utf-8"))
                                total += line_bytes
                                buffer_bytes += line_bytes
                                if buffer_bytes >= chunk_size:
                                    flush_buffer()
                        return True
                    except Exception:
                        log.exception("failed to read %s even with errors ignored", path)
                        return False

            if not process_file():
                continue
            first = False
            file_state[key] = {"hash": h, "size": size}

    flush_buffer()
    save_state(file_state)
    data = "".join(data_parts)
    if total >= threshold:
        temp_path.replace(dataset_path)
        return True, data
    temp_path.unlink(missing_ok=True)
    return False, data


def run_orchestrator(
    threshold: int = DEFAULT_THRESHOLD,
    dataset_dir: Path = CONFIG_DATASET_DIR,
    resume: bool = False,
    dry_run: bool = False,
    allow_ext: Iterable[str] = DEFAULT_ALLOW_EXT,
    deny_ext: Optional[Iterable[str]] = None,
) -> dict:
    """Collect data, train if ready, and return entropy metrics."""

    repo_root = Path(__file__).resolve().parents[1]
    base_paths = [repo_root / "artefacts", repo_root]
    ready, text = collect_new_data(
        base_paths,
        DATASET_FILE,
        threshold,
        resume=resume,
        allow_ext=allow_ext,
        deny_ext=deny_ext,
    )
    metrics = {
        "markov_entropy": round(markov_entropy(text), 2),
        "model_perplexity": round(model_perplexity(text), 2),
    }
    if ready and not dry_run:
        prepare_char_dataset(text, dataset_dir)
        train_model(dataset_dir, Path(__file__).parent / "weights")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dataset_dir", type=str, default=str(CONFIG_DATASET_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metric", choices=["markov", "neural", "both"], default="both")
    parser.add_argument(
        "--allow-ext",
        action="append",
        dest="allow_ext",
        help="Allowed file extensions (can be repeated)",
    )
    parser.add_argument(
        "--deny-ext",
        action="append",
        dest="deny_ext",
        help="Disallowed file extensions (can be repeated)",
    )
    args = parser.parse_args()

    metrics = run_orchestrator(
        threshold=args.threshold,
        dataset_dir=Path(args.dataset_dir),
        resume=args.resume,
        dry_run=args.dry_run,
        allow_ext=args.allow_ext if args.allow_ext is not None else DEFAULT_ALLOW_EXT,
        deny_ext=args.deny_ext,
    )
    if args.metric == "markov":
        out = {"markov_entropy": metrics["markov_entropy"]}
    elif args.metric == "neural":
        out = {"model_perplexity": metrics["model_perplexity"]}
    else:
        out = metrics
    LOGGER.info(json.dumps(out))


if __name__ == "__main__":
    main()
