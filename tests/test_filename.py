import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.tools import sanitize_filename  # noqa: E402


def test_sanitize_filename_removes_traversal():
    assert sanitize_filename('../secret.txt') == 'secret.txt'
