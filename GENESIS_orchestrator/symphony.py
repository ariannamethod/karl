import argparse
import json
import logging
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Tuple

from . import state
from .config import dataset_dir as CONFIG_DATASET_DIR, model_hyperparams, threshold as DEFAULT_THRESHOLD

LOGGER = logging.getLogger(__name__)
DATASET_FILE = Path(__file__).with_name('gen_data.txt')

def _looks_binary(path: Path) -> bool:
    try:
        with path.open('rb') as f:
            chunk = f.read(1024)
        return b'\0' in chunk
    except Exception:
        return True

def _iter_text_files(paths: Iterable[Path], exclude: Iterable[Path]) -> Iterable[Path]:
    exclude = {p.resolve() for p in exclude}
    for base in paths:
        for path in base.rglob('*'):
            if path.is_file() and path.resolve() not in exclude and not _looks_binary(path):
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
                h = state.file_hash(path)
                size = path.stat().st_size
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

def _prepare_char_dataset(text: str, dest: Path) -> None:
    import pickle
    import numpy as np
    chars = sorted(set(text))
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
        LOGGER.warning('dataset not found, skipping training')
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

    LOGGER.info('starting training; device=%s', device)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True,
            text=True,
        )
        LOGGER.info('training stdout:\n%s', result.stdout)
        if result.stderr:
            LOGGER.warning('training stderr:\n%s', result.stderr)
        duration = time.time() - start
        LOGGER.info('training completed in %.2f seconds', duration)
    except subprocess.CalledProcessError as exc:
        duration = time.time() - start
        LOGGER.error(
            'training failed after %.2f seconds\nstdout:\n%s\nstderr:\n%s',
            duration,
            exc.stdout,
            exc.stderr,
        )
    except Exception:
        duration = time.time() - start
        LOGGER.exception('training failed after %.2f seconds', duration)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument('--dataset_dir', type=str, default=str(CONFIG_DATASET_DIR))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_paths = [repo_root / 'artefacts', repo_root]
    ready, text = collect_new_data(base_paths, DATASET_FILE, args.threshold, resume=args.resume)
    entropy = markov_entropy(text)
    print(json.dumps({'markov_entropy': round(entropy, 2)}))
    if ready and not args.dry_run:
        dataset_dir = Path(args.dataset_dir)
        _prepare_char_dataset(text, dataset_dir)
        train_model(dataset_dir, Path(__file__).parent / 'weights')

if __name__ == '__main__':
    main()
