"""Root conftest.py — adds project root to sys.path so 'scripts' and 'src' are importable."""
import sys
import os

# Ensure the project root (this file's directory) is on sys.path
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
