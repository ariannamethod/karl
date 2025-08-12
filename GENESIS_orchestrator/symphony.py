"""Data collection and training utilities.

The module filters processed files by extension and avoids hashing very large
files to reduce resource usage.
"""

import argparse
import json
import logging
import math
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Tuple

from . import state
from .config import dataset_dir as CONFIG_DATASET_DIR, model_hyperparams, threshold as DEFAULT_THRESHOLD

LOGGER = logging.getLogger(__name__)
DATASET_FILE = Path(__file__).with_name('gen_data.txt')

# Default set of file extensions considered for processing.
DEFAULT_ALLOW_EXT = {".txt", ".md", ".py"}

def _looks_binary(path: Path) -> bool:
    try:
        with path.open('rb') as f:
            chunk = f.read(1024)
        return b'\0' in chunk
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
        for path in base.rglob('*'):
            if not path.is_file() or path.resolve() in exclude or _looks_binary(path):
                continue
            suffix = path.suffix.lower()
            if allow is not None and suffix not in allow:
                continue
            if suffix in deny:
                continue
            yield path

def collect_new_data(base_paths: Iterable[Path], dataset_path: Path = DATASET_FILE,
                     threshold: int = DEFAULT_THRESHOLD, resume: bool = False,
                     logger: Optional[logging.Logger] = None) -> Tuple[bool, str]:
    """Collect text from base_paths and write to dataset_path when threshold exceeded.

    When ``resume`` is True, previously processed files are skipped based on stored
    hashes and sizes.
    """
    log = logger or LOGGER
    file_state = state.load_state() if resume else {}
    temp_path = dataset_path.with_suffix(dataset_path.suffix + '.tmp')
    total = 0
    parts = []
    with temp_path.open('w', encoding='utf-8') as tmp:
        first = True
        for path in _iter_text_files(base_paths, exclude=[dataset_path, state.STATE_FILE, temp_path]):
            try:
                size = path.stat().st_size
                h = state.file_hash(path, size=size)
                if h is None:
                    continue
            except Exception:
                log.exception("failed to hash %s", path)
                continue
            key = str(path.resolve())
            entry = file_state.get(key)
            if entry and entry.get('hash') == h and entry.get('size') == size:
                continue
            try:
                text = path.read_text(encoding='utf-8')
            except Exception:
                log.exception("failed to read %s", path)
                try:
                    text = path.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    log.exception("failed to read %s even with errors ignored", path)
                    continue
            if not first:
                tmp.write('\n')
            tmp.write(text)
            first = False
            parts.append(text)
            total += len(text.encode('utf-8'))
            file_state[key] = {'hash': h, 'size': size}
    state.save_state(file_state)
    data = ''.join(parts)
    if total >= threshold:
        temp_path.replace(dataset_path)
        return True, data
    temp_path.unlink(missing_ok=True)
    return False, data

def markov_entropy(text: str, n: int = 2) -> float:
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    if not text:
        return 0.0
    n = max(1, min(n, len(text)))
    counts = Counter(text[i:i + n] for i in range(len(text) - n + 1))
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def model_perplexity(text: str) -> float:
    """Return the perplexity of ``text`` under the trained mini-GPT model.

    The function loads the checkpoint produced by :func:`train_model` from the
    ``weights`` directory and evaluates the cross-entropy loss on the provided
    ``text``. Perplexity is ``exp(loss)``. If no model weights are available or
    the text is too short, ``0.0`` is returned.
    """

    weights_path = Path(__file__).with_name("weights") / "model.pth"
    if not text or not weights_path.exists():
        return 0.0

    try:
        import torch

        checkpoint = torch.load(weights_path, map_location="cpu")
    except Exception:
        return 0.0

    model_args = checkpoint.get("model_args")
    if not model_args:
        return 0.0

    from .model import GPT, GPTConfig

    model = GPT(GPTConfig(**model_args))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    try:
        with open(Path(CONFIG_DATASET_DIR) / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        stoi = meta["stoi"]
    except Exception:
        return 0.0

    encoded = [stoi.get(ch, 0) for ch in text]
    if len(encoded) < 2:
        return 0.0
    ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        _, loss = model(ids[:, :-1], ids[:, 1:])
    return float(math.exp(loss.item()))

def _prepare_char_dataset(text: str, dest: Path) -> None:
    import pickle
    import numpy as np

    chars = sorted(set(text))
    if not text or not chars:
        raise ValueError("text must be non-empty and contain at least one unique character")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    n = len(text)
    train_data = text[: int(n * 0.9)]
    val_data = text[int(n * 0.9):]
    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)
    dest.mkdir(parents=True, exist_ok=True)
    train_ids.tofile(dest / 'train.bin')
    val_ids.tofile(dest / 'val.bin')
    meta = {'vocab_size': len(chars), 'itos': itos, 'stoi': stoi}
    with open(dest / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

def train_model(dataset_dir: Path, out_dir: Path) -> None:
    if not dataset_dir.exists():
        print('dataset not found, skipping training')
        return

    # determine device; default to cpu if torch is missing
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'

    weights_dir = out_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        'train.py',
        f'--dataset={dataset_dir}',
        f'--device={device}',
        '--compile=False',
        '--eval_iters=1',
        '--log_interval=1',
    ]

    hyperparams = dict(model_hyperparams)
    if device == 'cpu':
        hyperparams.update({
            'block_size': 32,
            'batch_size': 4,
            'n_layer': 1,
            'n_head': 1,
            'n_embd': 32,
        })
    for key, value in hyperparams.items():
        cmd.append(f'--{key}={value}')
    cmd.append(f'--out_dir={weights_dir}')
    try:
        subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        ckpt = weights_dir / 'ckpt.pt'
        if ckpt.exists():
            shutil.copy(ckpt, weights_dir / 'model.pth')
    except Exception as exc:
        print(f'training failed: {exc}')


def run_orchestrator(
    threshold: int = DEFAULT_THRESHOLD,
    dataset_dir: Path = CONFIG_DATASET_DIR,
    resume: bool = False,
    dry_run: bool = False,
) -> dict:
    """Collect data, train if ready, and return entropy metrics."""

    repo_root = Path(__file__).resolve().parents[1]
    base_paths = [repo_root / 'artefacts', repo_root]
    ready, text = collect_new_data(base_paths, DATASET_FILE, threshold, resume=resume)
    metrics = {
        'markov_entropy': round(markov_entropy(text), 2),
        'model_perplexity': round(model_perplexity(text), 2),
    }
    if ready and not dry_run:
        _prepare_char_dataset(text, dataset_dir)
        train_model(dataset_dir, Path(__file__).parent / 'weights')
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument('--dataset_dir', type=str, default=str(CONFIG_DATASET_DIR))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--metric', choices=['markov', 'neural', 'both'], default='both'
    )
    args = parser.parse_args()

    metrics = run_orchestrator(
        threshold=args.threshold,
        dataset_dir=Path(args.dataset_dir),
        resume=args.resume,
        dry_run=args.dry_run,
    )
    if args.metric == 'markov':
        out = {'markov_entropy': metrics['markov_entropy']}
    elif args.metric == 'neural':
        out = {'model_perplexity': metrics['model_perplexity']}
    else:
        out = metrics
    print(json.dumps(out))

if __name__ == '__main__':
    main()
