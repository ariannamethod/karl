import argparse
import json
import math
import os
import sys
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

ALLOWED_EXTS = {'.txt', '.md', '.log', '.jsonl'}
DATASET_FILE = Path(__file__).with_name('gen_data.txt')
DEFAULT_THRESHOLD = 256 * 1024  # 256KB

def _iter_text_files(paths: Iterable[Path]) -> Iterable[Path]:
    for base in paths:
        for path in base.rglob('*'):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTS:
                yield path

def collect_new_data(base_paths: Iterable[Path], dataset_path: Path = DATASET_FILE,
                     threshold: int = DEFAULT_THRESHOLD) -> Tuple[bool, str]:
    """Collect text from base_paths and write to dataset_path when threshold exceeded."""
    collected = []
    total = 0
    for path in _iter_text_files(base_paths):
        try:
            text = path.read_text(encoding='utf-8')
        except Exception:
            text = path.read_text(encoding='utf-8', errors='ignore')
        collected.append(text)
        total += len(text.encode('utf-8'))
    if total >= threshold:
        dataset_path.write_text('\n'.join(collected), encoding='utf-8')
        return True, ''.join(collected)
    return False, ''.join(collected)

def markov_entropy(text: str, n: int = 2) -> float:
    if len(text) < n:
        return 0.0
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
        print('dataset not found, skipping training')
        return
    weights_dir = out_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        'train.py',
        f'--dataset={dataset_dir.name}',
        '--device=cpu',
        '--compile=False',
        '--eval_iters=1',
        '--log_interval=1',
        '--block_size=64',
        '--batch_size=12',
        '--n_layer=2',
        '--n_head=2',
        '--n_embd=64',
        '--max_iters=10',
        '--lr_decay_iters=10',
        '--dropout=0.0',
        f'--out_dir={weights_dir}',
    ]
    try:
        subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
    except Exception as exc:
        print(f'training failed: {exc}')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_paths = [repo_root / 'artefacts', repo_root]
    ready, text = collect_new_data(base_paths, DATASET_FILE, args.threshold)
    entropy = markov_entropy(text)
    print(json.dumps({'markov_entropy': round(entropy, 2)}))
    if ready and not args.dry_run:
        dataset_dir = Path(__file__).parent / 'data' / 'genesis'
        _prepare_char_dataset(text, dataset_dir)
        train_model(dataset_dir, Path(__file__).parent / 'weights')

if __name__ == '__main__':
    main()
