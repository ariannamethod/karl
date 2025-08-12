import json
import logging
import re
from pathlib import Path

LOG_FILE = Path("artefacts/blocked_commands.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
ALLOW_FILE = Path("artefacts/allowed_commands.json")

logger = logging.getLogger("security")
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BLOCK_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"sudo\s",
    r":\(\)\s*{\s*:\|:&\s*};\s*:",
    r"curl\s",
    r"wget\s",
    r"ssh\s",
    r"scp\s",
    r"nc\s",
    r"netcat\s",
    r"telnet\s",
    r"ping\s",
    r"apt(-get)?\s",
    r"pip\s+install",
    r"python\s+-m\s+http\.server",
    r">/dev/tcp",
    r"xmrig",
    r"minerd",
]
BLOCK_REGEXES = [re.compile(p, re.IGNORECASE) for p in BLOCK_PATTERNS]

def is_blocked(command: str) -> bool:
    return any(regex.search(command) for regex in BLOCK_REGEXES)


def _load_allowlist() -> list[str]:
    """Return the list of allowed commands from ``ALLOW_FILE``.

    The file contains a JSON array. If the file is missing or empty, the
    function returns an empty list which indicates that all commands are
    allowed. Errors during parsing are silently ignored, yielding an empty
    allowlist.
    """

    try:
        data = json.loads(ALLOW_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(cmd) for cmd in data]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        logger.error("Failed to parse allowlist: %s", ALLOW_FILE)
    return []


def is_allowed(command: str) -> bool:
    """Check whether ``command`` is present in the allowlist.

    When the allowlist is empty or missing, all commands are permitted.
    Matching is performed by prefix, allowing allowlisted commands to contain
    arguments.
    """

    allowed = _load_allowlist()
    if not allowed:
        return True
    stripped = command.strip()
    return any(stripped.startswith(item) for item in allowed)

def log_blocked(command: str) -> None:
    logger.warning("Blocked command: %s", command)

__all__ = ["is_blocked", "is_allowed", "log_blocked"]
