#!/usr/bin/env python3
"""
Image + driving video demo launcher (wraps ``fom.demo``).

Run from project root; this directory is added to ``sys.path`` automatically.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fom.demo import main as demo_main


if __name__ == "__main__":
    demo_main()
