"""Lightweight interface for the GENESIS orchestrator.

Provides helper functions for updating the dataset, training the
mini-model/Markov chain, and reporting the latest entropy value.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from .symphony import run_orchestrator
from .orchestrator import STATE_FILE, threshold as DEFAULT_THRESHOLD

ENTROPY_FILE: Final[Path] = Path(__file__).with_name("last_entropy.json")
_last_entropy: float = 0.0
_status: str = ""


def _read_entropy_file() -> float:
    try:
        return float(json.loads(ENTROPY_FILE.read_text()).get("markov_entropy", 0.0))
    except Exception:
        return 0.0


def _write_entropy_file(value: float) -> None:
    try:
        ENTROPY_FILE.write_text(json.dumps({"markov_entropy": value}))
    except Exception:
        pass


_last_entropy = _read_entropy_file()


def update_and_train() -> None:
    """Collect new repository data and retrain if needed.

    On the very first run (when no state file exists) the entire repository is
    ingested and training is triggered immediately.
    """
    global _last_entropy, _status
    state_exists = STATE_FILE.exists()
    try:
        metrics = run_orchestrator(
            threshold=0 if not state_exists else DEFAULT_THRESHOLD,
            resume=state_exists,
        )
        _last_entropy = float(metrics.get("markov_entropy", 0.0))
        _write_entropy_file(_last_entropy)
        _status = "⏳"
    except Exception:
        _status = ""
        raise


def report_entropy() -> float:
    """Return the most recently computed Markov entropy."""
    return _last_entropy


def status_emoji() -> str:
    """Return an hourglass emoji when a training update occurred.

    The emoji is returned once per update and then cleared so subsequent calls
    yield an empty string.
    """
    global _status
    emoji = _status
    _status = ""
    return emoji
