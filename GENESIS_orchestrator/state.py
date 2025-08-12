import json
import hashlib
from pathlib import Path
from typing import Dict, Any

STATE_FILE = Path(__file__).with_name('state.json')

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state))

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
