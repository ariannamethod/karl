import logging
import re
from pathlib import Path

LOG_FILE = Path("artefacts/blocked_commands.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

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

def log_blocked(command: str) -> None:
    logger.warning("Blocked command: %s", command)

__all__ = ["is_blocked", "log_blocked"]
