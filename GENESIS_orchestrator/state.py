"""State persistence and file hashing utilities.

This module stores metadata about processed files and exposes :func:`file_hash`
which skips hashing files that exceed a configurable size limit.
"""

import json
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

STATE_FILE = Path(__file__).with_name('state.json')
STATE_VERSION = 1

# Default maximum file size (in bytes) for hashing. Can be overridden via the
# ``FILE_HASH_MAX_SIZE`` environment variable.
MAX_HASH_SIZE = int(os.environ.get("FILE_HASH_MAX_SIZE", 5 * 1024 * 1024))

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
    tmp_path = STATE_FILE.with_suffix('.tmp')
    data = {'version': STATE_VERSION, 'files': state}
    try:
        tmp_path.write_text(json.dumps(data))
        tmp_path.replace(STATE_FILE)
    except Exception as exc:
        logger.error("failed to save state file %s: %s", STATE_FILE, exc)
        try:
            tmp_path.unlink()
        except Exception:
            pass

def file_hash(path: Path, max_size: Optional[int] = MAX_HASH_SIZE, *, size: Optional[int] = None) -> Optional[str]:
    """Compute the SHA256 hash of ``path``.

    If ``max_size`` is provided and the file exceeds that size in bytes, the
    function returns ``None`` instead of hashing to avoid the cost of reading
    very large files.
    """
    actual_size = size if size is not None else path.stat().st_size
    if max_size is not None and actual_size > max_size:
        logger.info("skipping hash for %s: size %s exceeds limit %s", path, actual_size, max_size)
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
