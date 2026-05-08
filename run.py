#!/usr/bin/env python3
"""
Alternate entry: same as ``python -m fom`` with ``PYTHONPATH`` set to this project root.

Usage from project root::

    python run.py --config config/mgif-256-local.yaml --mode train --log_dir logs
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fom.entry import main as entry_main


if __name__ == "__main__":
    entry_main()
