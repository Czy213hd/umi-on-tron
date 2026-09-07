#!/usr/bin/env python3
"""Aggregate batched ablation rollouts into per-seed and mean ± std tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--num-targets", type=int, required=True)
parser.add_argument("--seeds", type=int, nargs="+", required=True)
args = parser.parse_args()

models = (
    "full_baseline",
    "no_latent",
    "dreamwaq_cenet",
    "matched_additive_reward",
    "privileged_oracle",
)
metrics = (
    "success_5cm_5deg_rate",
    "position_mean_m",
    "position_median_m",
    "position_p95_m",
    "orientation_mean_rad",
    "orientation_p95_rad",
    "termination_rate",
    "action_rate_l2_mean",
    "action_rate_l2_p95",
    "action_rms_mean",
)


def load_seed(model: str, seed: int) -> tuple[dict[str, float | int | str], list[tuple[float, ...]]]:
    seed_dir = args.input_dir / model / f"seed_{seed}"
    paths = sorted(seed_dir.glob("batch_*.csv"), key=lambda path: int(path.stem.split("_")[-1]))
    samples: list[dict[str, str]] = []
    target_rows: list[tuple[float, ...]] = []
    for path in paths:
        offset = int(path.stem.split("_")[-1])
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["global_target_id"] = str(offset + int(row["env_id"]))
            samples.append(row)
    samples.sort(key=lambda row: int(row["global_target_id"]))
    if len(samples) != args.num_targets:
        raise RuntimeError(f"{model} seed {seed}: expected {args.num_targets} rows, got {len(samples)}")
    if [int(row["global_target_id"]) for row in samples] != list(range(args.num_targets)):
        raise RuntimeError(f"{model} seed {seed}: target ids are incomplete or duplicated")

    target_columns = ("target_x", "target_y", "target_z", "target_roll", "target_pitch", "target_yaw")
    target_rows = [tuple(float(row[column]) for column in target_columns) for row in samples]
    pos = np.asarray([float(row["final_pos_mean_m"]) for row in samples])
    ori = np.asarray([float(row["final_ori_mean_rad"]) for row in samples])
    success = np.asarray([int(row["success_5cm_5deg"]) for row in samples])
    terminated = np.asarray([int(row["terminated"]) for row in samples])
    action_rate = np.asarray([float(row["action_rate_l2_mean"]) for row in samples])
    action_rms = np.asarray([float(row["action_rms"]) for row in samples])
    finite_pos = pos[np.isfinite(pos)]
    finite_ori = ori[np.isfinite(ori)]
    finite_rate = action_rate[np.isfinite(action_rate)]
    finite_rms = action_rms[np.isfinite(action_rms)]
    result: dict[str, float | int | str] = {
        "model": model,
        "seed": seed,
        "num_targets": len(samples),
        "success_5cm_5deg_rate": float(success.mean()),
        "position_mean_m": float(finite_pos.mean()),
        "position_median_m": float(np.median(finite_pos)),
        "position_p95_m": float(np.quantile(finite_pos, 0.95)),
        "orientation_mean_rad": float(finite_ori.mean()),
        "orientation_p95_rad": float(np.quantile(finite_ori, 0.95)),
        "termination_rate": float(terminated.mean()),
        "action_rate_l2_mean": float(finite_rate.mean()),
        "action_rate_l2_p95": float(np.quantile(finite_rate, 0.95)),
        "action_rms_mean": float(finite_rms.mean()),
    }
    return result, target_rows


per_seed: list[dict[str, float | int | str]] = []
targets_by_seed: dict[int, list[tuple[float, ...]]] = {}
for model in models:
    for seed in args.seeds:
        result, targets = load_seed(model, seed)
        if seed in targets_by_seed and targets != targets_by_seed[seed]:
            raise RuntimeError(f"Target mismatch for {model}, seed {seed}")
        targets_by_seed.setdefault(seed, targets)
        per_seed.append(result)

per_seed_path = args.input_dir / "per_seed.csv"
with per_seed_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
    writer.writeheader()
    writer.writerows(per_seed)

summary_rows = []
for model in models:
    model_rows = [row for row in per_seed if row["model"] == model]
    summary: dict[str, float | int | str] = {
        "model": model,
        "num_seeds": len(model_rows),
        "targets_per_seed": args.num_targets,
    }
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in model_rows])
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1))
    summary_rows.append(summary)

summary_csv = args.input_dir / "summary_mean_std.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)

display_metrics = (
    ("success_5cm_5deg_rate", "Success 5cm/5deg"),
    ("position_mean_m", "Position mean (m)"),
    ("position_p95_m", "Position P95 (m)"),
    ("orientation_mean_rad", "Orientation mean (rad)"),
    ("termination_rate", "Termination"),
    ("action_rate_l2_mean", "Action rate L2"),
    ("action_rms_mean", "Action RMS"),
)
lines = [
    "| Model | " + " | ".join(title for _, title in display_metrics) + " |",
    "| --- | " + " | ".join("---:" for _ in display_metrics) + " |",
]
for row in summary_rows:
    values = []
    for metric, _ in display_metrics:
        mean = float(row[f"{metric}_mean"])
        std = float(row[f"{metric}_std"])
        values.append(f"{mean:.5f} ± {std:.5f}")
    lines.append(f"| {row['model']} | " + " | ".join(values) + " |")
summary_md = args.input_dir / "summary_mean_std.md"
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

metadata = {
    "seeds": args.seeds,
    "targets_per_seed": args.num_targets,
    "total_rollouts_per_model": args.num_targets * len(args.seeds),
    "paired_targets_within_seed": True,
    "aggregation": "compute each metric per seed, then report sample mean and sample std across seeds",
}
(args.input_dir / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
