import json
import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

STATE_FILE = Path(__file__).with_name('state.json')
STATE_VERSION = 1

logger = logging.getLogger(__name__)

def _migrate_state(data: Dict[str, Any], version: int) -> Dict[str, Any]:
    """Migrate legacy state formats to the current structure.

    Version ``0`` stored the file mapping directly without a version key.
    """
    if version == 0:
        return data if isinstance(data, dict) else {}
    logger.warning("no migration path for state version %s", version)
    return {}

def load_state() -> Dict[str, Any]:
    """Load the state file.

    Returns the mapping of file paths to hash/size information. If the file is
    missing, unreadable or incompatible, an empty mapping is returned.
    """
    if not STATE_FILE.exists():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text())
    except Exception as exc:
        logger.error("failed to read state file %s: %s", STATE_FILE, exc)
        return {}
    if isinstance(data, dict) and 'version' in data:
        version = data.get('version', 0)
        if version == STATE_VERSION:
            files = data.get('files', {})
            return files if isinstance(files, dict) else {}
        if version < STATE_VERSION:
            try:
                return _migrate_state(data, version)
            except Exception as exc:
                logger.error("failed to migrate state: %s", exc)
                return {}
        logger.warning("unsupported state version %s", version)
        return {}
    return data if isinstance(data, dict) else {}

def save_state(state: Dict[str, Any]) -> None:
    """Persist state atomically.

    The state is wrapped with a version header and written to a temporary file
    before being atomically renamed into place.
    """
    data = {'version': STATE_VERSION, 'files': state}
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=STATE_FILE.parent, delete=False) as tmp_file:
            json.dump(data, tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            tmp_path = Path(tmp_file.name)
        os.replace(tmp_path, STATE_FILE)
    except (IOError, OSError) as exc:
        logger.error("failed to save state file %s: %s", STATE_FILE, exc)
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
