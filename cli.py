#!/usr/bin/env python3
"""
Thin launcher: prepend this project root to ``sys.path`` and forward to ``fom.entry``.

You can call modes as the first word (``train``, ``reconstruction``, ``animate``) or
pass ``--mode`` explicitly. If ``--config`` is omitted, ``config/mgif-256.yaml`` is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "mgif-256.yaml"
_MODE_ALIASES = frozenset({"train", "reconstruction", "animate"})


def main() -> None:
    sys.path.insert(0, str(_PROJECT_ROOT))
    argv = sys.argv[1:]
    if argv and argv[0] in _MODE_ALIASES:
        argv = ["--mode", argv[0]] + argv[1:]
    if "--config" not in argv:
        argv = ["--config", str(_DEFAULT_CONFIG)] + argv
    from fom.entry import main as run_entrypoint

    run_entrypoint(argv)


if __name__ == "__main__":
    main()
