"""Default configuration for the GENESIS orchestrator.

This module provides basic settings used by the orchestration scripts. The
values defined here can be overridden at runtime via command line arguments.
"""

from pathlib import Path

# Size threshold for collected data before triggering training (in bytes)
threshold: int = 256 * 1024  # 256KB

# Directory where the prepared dataset will be stored
dataset_dir: Path = Path(__file__).parent / "data" / "genesis"

# Hyperparameters for the training script. These values map directly to the
# command line options of ``train.py`` and can be selectively overridden.
model_hyperparams = {
    "block_size": 64,
    "batch_size": 12,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
    "max_iters": 10,
    "lr_decay_iters": 10,
    "dropout": 0.0,
}

