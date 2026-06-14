"""Train a Matlas mjlab task with RSL-RL PPO.

Thin wrapper around mjlab's training CLI: importing ``matlas.tasks.stand``
registers our tasks into mjlab's shared registry before its ``main()`` lists
available tasks. Run from the repo root, e.g.::

    uv run python scripts/train.py Mjlab-Matlas-Stand --env.scene.num-envs 4096

Training requires a CUDA GPU (MuJoCo-Warp). On CPU/macOS it falls back to a
slow single-device run useful only for smoke tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matlas.tasks  # noqa: F401  (registers all tasks as a side effect)
from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
