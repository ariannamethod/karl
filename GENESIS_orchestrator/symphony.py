import argparse
import json
import logging
import math
import sys
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import yaml

DATASET_FILE = Path(__file__).with_name('gen_data.txt')
DEFAULT_THRESHOLD = 256 * 1024  # 256KB


@dataclass
class TrainerConfig:
    compile: bool = False
    eval_iters: int = 1
    log_interval: int = 1
    block_size: int = 64
    batch_size: int = 12
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    max_iters: int = 10
    lr_decay_iters: int = 10
    dropout: float = 0.0
    out_dir: str = 'weights'

    @classmethod
    def from_yaml(cls, path: Path) -> 'TrainerConfig':
        with path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


def _detect_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'

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
                     threshold: int = DEFAULT_THRESHOLD) -> Tuple[bool, str]:
    """Collect text from base_paths and write to dataset_path when threshold exceeded."""
    collected = []
    total = 0
    for path in _iter_text_files(base_paths, exclude=[dataset_path]):
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

def train_model(dataset_dir: Path, config: TrainerConfig) -> None:
    if not dataset_dir.exists():
        print('dataset not found, skipping training')
        return
    device = _detect_device()
    weights_dir = Path(__file__).parent / config.out_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        'train.py',
        f'--dataset={dataset_dir.name}',
        f'--device={device}',
        f'--compile={str(config.compile)}',
        f'--eval_iters={config.eval_iters}',
        f'--log_interval={config.log_interval}',
        f'--block_size={config.block_size}',
        f'--batch_size={config.batch_size}',
        f'--n_layer={config.n_layer}',
        f'--n_head={config.n_head}',
        f'--n_embd={config.n_embd}',
        f'--max_iters={config.max_iters}',
        f'--lr_decay_iters={config.lr_decay_iters}',
        f'--dropout={config.dropout}',
        f'--out_dir={weights_dir}',
    ]
    logger = logging.getLogger('genesis_train')
    if not logger.handlers:
        logs_dir = Path(__file__).resolve().parents[1] / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(logs_dir / 'genesis_train.log')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        result.check_returncode()
    except Exception as exc:
        logger.exception('training failed: %s', exc)
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
        config_path = Path(__file__).resolve().parents[1] / 'config' / 'genesis.yaml'
        config = TrainerConfig.from_yaml(config_path)
        train_model(dataset_dir, config)

if __name__ == '__main__':
    main()
