"""
conftest.py
-----------
Pytest automatically loads this file before running any tests in this
folder. We use it to add src/ to Python's import path, so test files
can write `from train import ...` / `from predict import ...` the same
way scripts inside src/ already do, without needing to package/install
the project first.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
