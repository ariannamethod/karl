"""Test fixtures."""

# flake8: noqa

import os
import sys
import pytest

@pytest.fixture(autouse=True)
def add_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, project_root)
    try:
        yield
    finally:
        if project_root in sys.path:
            while project_root in sys.path:
                sys.path.remove(project_root)
