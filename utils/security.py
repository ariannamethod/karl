import logging
import re
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOG_FILE = Path("artefacts/blocked_commands.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("security")
if not logger.handlers:
    handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", backupCount=7, encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Whitelist of allowed commands
ALLOWED_PATTERNS = [
    r"^echo\b",
    r"^ls\b",
    r"^cat\b",
    r"^pwd\b",
    r"^whoami\b",
    r"^date\b",
]
ALLOWED_REGEXES = [re.compile(p, re.IGNORECASE) for p in ALLOWED_PATTERNS]

# Suspicious sequences to warn about
SUSPICIOUS_PATTERNS = [
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
SUSPICIOUS_REGEXES = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]


def is_blocked(command: str) -> bool:
    """Return True if command is not in the whitelist.

    Suspicious sequences are logged regardless of allow status.
    """

    if any(regex.search(command) for regex in SUSPICIOUS_REGEXES):
        logger.warning("Suspicious command sequence: %s", command)

    allowed = any(regex.search(command) for regex in ALLOWED_REGEXES)
    return not allowed


def log_blocked(command: str) -> None:
    """Log a blocked command attempt."""

    logger.error("Blocked command: %s", command)


__all__ = ["is_blocked", "log_blocked"]
