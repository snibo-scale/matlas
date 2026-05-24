"""
Sweep harness for matlas standing-policy experiments.

Subcommands:
  list       Print available sweeps.
  plan       Print the commands a sweep would run, without executing.
  run        Execute a sweep (skips cells that already have final.zip).
  aggregate  Read evaluations.npz from each completed cell, write summary CSV.

Sweeps are defined in the SWEEPS dict near the bottom of this file. Each cell
is `(axis combo) x seed`; per-cell results land in:

    runs/sweeps/<sweep>/<cell_id>/seed_<k>/
        stage_<n>.zip
        stage_<n>_eval/evaluations.npz   (only if --eval-freq > 0)
        stage_<n>_best/                  (best model from EvalCallback)
        final.zip
        log.txt

Example:
    uv run scripts/sweep.py list
    uv run scripts/sweep.py plan --sweep exploration
    uv run scripts/sweep.py run  --sweep exploration --workers 4
    uv run scripts/sweep.py aggregate --sweep exploration
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SWEEPS_DIR = ROOT / "runs" / "sweeps"
TRAINER = ROOT / "scripts" / "train_stand_stages.py"

# Canonical stage-1 warm-start. Trained with:
#   uv run scripts/train_stand_stages.py --algo ppo --stages 1 \
#     --timesteps-per-stage 250000 --seed 1 \
#     --out-dir runs/diag/v3_stage1_ppo_long --eval-freq 10000
# Best checkpoint: ep_len=144 / max 300 under deterministic eval, return=461.7.
# Beat SAC head-to-head on stability (no eval crashes) and matched peak.
# LR decay in the trainer (--lr-schedule linear) is what enabled the long run
# to keep improving instead of collapsing past 200k like the original config did.
STAGE1_WARM_START = ROOT / "runs" / "diag" / "v3_stage1_ppo_long" / "stage_1_best" / "best_model.zip"


# ---------------------------------------------------------------------------
# Sweep spec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SweepSpec:
    name: str
    description: str
    axes: dict[str, list[Any]] = dataclasses.field(default_factory=dict)
    fixed_args: dict[str, Any] = dataclasses.field(default_factory=dict)
    stage_variants: list[list[int]] | None = None
    stages: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])
    timesteps_per_stage: int = 50_000
    seeds: list[int] = dataclasses.field(default_factory=lambda: [1, 2, 3])
    metric_stage: int = 3
    eval_freq: int = 5_000
    eval_episodes: int = 5
    # Optional path to a saved PPO/SAC model to warm-start every cell from.
    # The trainer's --load-model swaps in this checkpoint before stage training.
    warm_start: Path | None = None
    algo: str = "ppo"  # "ppo" or "sac"


# Map sweep-axis names to the CLI flags exposed by train_stand_stages.py.
# If you need to sweep a parameter not listed here, add the matching --flag to
# train_stand_stages.py first, then register it below.
AXIS_TO_FLAG: dict[str, str] = {
    "log_std_init":             "--log-std-init",
    "ent_coef":                 "--ent-coef",
    "learning_rate":            "--learning-rate",
    "clip_range":               "--clip-range",
    "vf_coef":                  "--vf-coef",
    "max_grad_norm":            "--max-grad-norm",
    "n_steps":                  "--n-steps",
    "batch_size":               "--batch-size",
    "gamma":                    "--gamma",
    "gae_lambda":               "--gae-lambda",
    "torque_cost":              "--torque-cost",
    "action_rate_cost":         "--action-rate-cost",
    "joint_speed_cost":         "--joint-speed-cost",
    "posture_weight":           "--posture-weight",
    "stillness_weight":         "--stillness-weight",
    "angular_stillness_weight": "--angular-stillness-weight",
    "upright_weight":           "--upright-weight",
    "reset_noise":              "--reset-noise",
    "episode_seconds":          "--episode-seconds",
    "xy_drift_weight":          "--xy-drift-weight",
    "foot_height_weight":       "--foot-height-weight",
}


# ---------------------------------------------------------------------------
# Sweep registry
# ---------------------------------------------------------------------------

SWEEPS: dict[str, SweepSpec] = {
    "noise_floor": SweepSpec(
        name="noise_floor",
        description="Sweep 1 — seed-only baseline (defaults, 5 seeds, full curriculum).",
        seeds=[1, 2, 3, 4, 5],
    ),
    "exploration": SweepSpec(
        name="exploration",
        description="Sweep 2 — log_std_init x ent_coef on stage 1 only.",
        axes={
            "log_std_init": [-5.0, -4.0, -3.0, -2.0],
            "ent_coef":     [0.0, 0.005, 0.02],
        },
        stages=[1],
        metric_stage=1,
    ),
    "upright_shape": SweepSpec(
        name="upright_shape",
        description="Sweep 3a — upright_weight on stage 2 (warm-started from stage 1).",
        axes={"upright_weight": [0.5, 1.0, 1.5, 2.5]},
        stages=[2],
        metric_stage=2,
        warm_start=STAGE1_WARM_START,
    ),
    "posture_stillness": SweepSpec(
        name="posture_stillness",
        description="Sweep 3b — posture_weight x stillness_weight on stage 2 (warm-started).",
        axes={
            "posture_weight":   [0.4, 1.2, 2.0],
            "stillness_weight": [0.1, 0.5, 1.5],
        },
        stages=[2],
        metric_stage=2,
        warm_start=STAGE1_WARM_START,
    ),
    "dormant_terms": SweepSpec(
        name="dormant_terms",
        description="Sweep 3c — re-enable xy_drift and foot_height on stage 2 (warm-started).",
        axes={
            "xy_drift_weight":    [0.0, 0.2, 0.5],
            "foot_height_weight": [0.0, 0.2, 0.5],
        },
        stages=[2],
        metric_stage=2,
        warm_start=STAGE1_WARM_START,
    ),
    "curriculum_subset": SweepSpec(
        name="curriculum_subset",
        description="Sweep 4a — is stage 1 necessary? Same total budget, dropped stages.",
        stage_variants=[[1, 2, 3], [2, 3], [3]],
        timesteps_per_stage=50_000,
        metric_stage=3,
    ),
}


# ---------------------------------------------------------------------------
# Cell generation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Cell:
    sweep: str
    cell_id: str
    extra_args: tuple[tuple[str, str], ...]
    stages: tuple[int, ...]
    seed: int

    @property
    def out_dir(self) -> Path:
        return SWEEPS_DIR / self.sweep / self.cell_id / f"seed_{self.seed}"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def generate_cells(spec: SweepSpec) -> list[Cell]:
    if spec.stage_variants is not None:
        stage_options = [
            (tuple(s), f"stages={'-'.join(str(x) for x in s)}")
            for s in spec.stage_variants
        ]
    else:
        stage_options = [(tuple(spec.stages), "")]

    axis_names = list(spec.axes.keys())
    axis_values = [spec.axes[k] for k in axis_names]
    axis_combos = list(itertools.product(*axis_values)) if axis_values else [()]

    cells: list[Cell] = []
    for stages, stage_tag in stage_options:
        for combo in axis_combos:
            parts: list[str] = []
            if stage_tag:
                parts.append(stage_tag)
            extra: list[tuple[str, str]] = []
            for name, value in zip(axis_names, combo):
                parts.append(f"{name}={_fmt(value)}")
                extra.append((AXIS_TO_FLAG[name], _fmt(value)))
            for name, value in spec.fixed_args.items():
                extra.append((AXIS_TO_FLAG[name], _fmt(value)))
            cell_id = "__".join(parts) if parts else "default"
            for seed in spec.seeds:
                cells.append(Cell(
                    sweep=spec.name,
                    cell_id=cell_id,
                    extra_args=tuple(extra),
                    stages=stages,
                    seed=seed,
                ))
    return cells


def build_command(cell: Cell, spec: SweepSpec) -> list[str]:
    cmd = [
        "uv", "run", str(TRAINER.relative_to(ROOT)),
        "--algo", spec.algo,
        "--stages", *[str(s) for s in cell.stages],
        "--timesteps-per-stage", str(spec.timesteps_per_stage),
        "--seed", str(cell.seed),
        "--out-dir", str(cell.out_dir),
        "--eval-freq", str(spec.eval_freq),
        "--eval-episodes", str(spec.eval_episodes),
    ]
    if spec.warm_start is not None:
        cmd.extend(["--load-model", str(spec.warm_start)])
    for flag, value in cell.extra_args:
        cmd.extend([flag, value])
    return cmd


def cell_complete(cell: Cell) -> bool:
    return (cell.out_dir / "final.zip").exists()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_one_cell(cell: Cell, spec: SweepSpec) -> tuple[Cell, int, Path]:
    cell.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell.out_dir / "log.txt"
    cmd = build_command(cell, spec)
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    return cell, proc.returncode, log_path


def cmd_plan(spec: SweepSpec) -> None:
    cells = generate_cells(spec)
    cell_ids = {c.cell_id for c in cells}
    print(f"# sweep={spec.name} cells={len(cell_ids)} runs={len(cells)}")
    print(f"# {spec.description}")
    for cell in cells:
        marker = "[skip]" if cell_complete(cell) else "[run] "
        print(f"{marker} {cell.cell_id} seed={cell.seed}")
        print("    " + " ".join(build_command(cell, spec)))


def cmd_run(spec: SweepSpec, workers: int, force: bool) -> None:
    if spec.warm_start is not None and not spec.warm_start.exists():
        raise SystemExit(
            f"Warm-start checkpoint missing: {spec.warm_start}\n"
            "Train it first (see STAGE1_WARM_START comment in sweep.py)."
        )
    cells = generate_cells(spec)
    todo = [c for c in cells if force or not cell_complete(c)]
    print(f"sweep={spec.name} total={len(cells)} todo={len(todo)} workers={workers}")
    if spec.warm_start is not None:
        print(f"  warm_start={spec.warm_start}")
    if not todo:
        print("nothing to do (all cells already have final.zip; pass --force to re-run)")
        return

    failed: list[tuple[Cell, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one_cell, c, spec): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            cell, rc, log_path = fut.result()
            status = "ok" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  [{i}/{len(todo)}] {status} {cell.cell_id} seed={cell.seed} -> {log_path}")
            if rc != 0:
                failed.append((cell, rc))

    if failed:
        print(f"\n{len(failed)} cells failed:")
        for cell, rc in failed:
            print(f"  rc={rc} {cell.out_dir}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_eval(cell: Cell, spec: SweepSpec) -> dict[str, float] | None:
    npz_path = cell.out_dir / f"stage_{spec.metric_stage}_eval" / "evaluations.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        results = data["results"]      # (n_evals, n_episodes_per_eval)
        ep_lens = data["ep_lengths"]   # (n_evals, n_episodes_per_eval)
    if results.size == 0:
        return None
    per_eval_return = results.mean(axis=1)
    per_eval_len = ep_lens.mean(axis=1)
    return {
        "final_return": float(per_eval_return[-1]),
        "best_return":  float(per_eval_return.max()),
        "final_ep_len": float(per_eval_len[-1]),
        "best_ep_len":  float(per_eval_len.max()),
        "n_evals":      int(results.shape[0]),
    }


def aggregate(spec: SweepSpec) -> list[dict[str, Any]]:
    cells = generate_cells(spec)
    by_cell: dict[str, list[Cell]] = {}
    for cell in cells:
        by_cell.setdefault(cell.cell_id, []).append(cell)

    rows: list[dict[str, Any]] = []
    for cell_id, cell_seeds in by_cell.items():
        seed_results = [load_eval(c, spec) for c in cell_seeds]
        complete = [r for r in seed_results if r is not None]
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "n_seeds_complete": len(complete),
            "n_seeds_total": len(cell_seeds),
        }
        if complete:
            finals = np.array([r["final_return"] for r in complete])
            bests = np.array([r["best_return"] for r in complete])
            ep_lens = np.array([r["final_ep_len"] for r in complete])
            row.update({
                "final_return_mean": float(finals.mean()),
                "final_return_std":  float(finals.std(ddof=0)),
                "final_return_min":  float(finals.min()),
                "final_return_max":  float(finals.max()),
                "best_return_mean":  float(bests.mean()),
                "final_ep_len_mean": float(ep_lens.mean()),
            })
        rows.append(row)

    rows.sort(key=lambda r: r.get("final_return_mean", -float("inf")), reverse=True)
    return rows


def cmd_aggregate(spec: SweepSpec, out_csv: Path) -> None:
    rows = aggregate(spec)
    if not rows:
        print("no rows; did you run the sweep?")
        return

    cols = [
        "cell_id", "n_seeds_complete", "n_seeds_total",
        "final_return_mean", "final_return_std", "final_return_min", "final_return_max",
        "best_return_mean", "final_ep_len_mean",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})
    print(f"wrote {out_csv}\n")

    widths = {c: max(len(c), max((len(_fmt(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for row in rows:
        print(" | ".join(_fmt(row.get(c, "")).ljust(widths[c]) for c in cols))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List available sweeps.")

    p_plan = sub.add_parser("plan", help="Print commands without running.")
    p_plan.add_argument("--sweep", required=True)

    p_run = sub.add_parser("run", help="Execute a sweep.")
    p_run.add_argument("--sweep", required=True)
    p_run.add_argument("--workers", type=int, default=1)
    p_run.add_argument("--force", action="store_true",
                       help="Re-run cells that already have final.zip.")

    p_agg = sub.add_parser("aggregate", help="Build a summary CSV from completed cells.")
    p_agg.add_argument("--sweep", required=True)
    p_agg.add_argument("--out-csv", type=Path, default=None)

    args = p.parse_args()

    if args.cmd == "list":
        for name, spec in SWEEPS.items():
            print(f"{name:20s} {spec.description}")
        return

    if args.sweep not in SWEEPS:
        raise SystemExit(f"Unknown sweep '{args.sweep}'. Available: {sorted(SWEEPS)}")
    spec = SWEEPS[args.sweep]

    if args.cmd == "plan":
        cmd_plan(spec)
    elif args.cmd == "run":
        cmd_run(spec, workers=args.workers, force=args.force)
    elif args.cmd == "aggregate":
        out = args.out_csv if args.out_csv else (SWEEPS_DIR / args.sweep / "summary.csv")
        cmd_aggregate(spec, out)


if __name__ == "__main__":
    main()
