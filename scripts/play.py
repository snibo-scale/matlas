"""Play / render a trained Matlas mjlab policy.

Wrapper around mjlab's play CLI; importing ``matlas.tasks.stand`` registers our
tasks first. Run from the repo root, e.g.::

    uv run python scripts/play.py Mjlab-Matlas-Stand --checkpoint-file <ckpt.pt>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matlas.tasks  # noqa: F401  (registers all tasks as a side effect)
from mjlab.scripts.play import main

if __name__ == "__main__":
    main()
