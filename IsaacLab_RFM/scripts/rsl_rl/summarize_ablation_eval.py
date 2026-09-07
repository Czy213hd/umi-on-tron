#!/usr/bin/env python3
"""Combine per-model ablation evaluation summaries into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--max-envs", type=int, default=None, help="Compare only env_id values below this limit.")
args = parser.parse_args()

paths = sorted(args.input_dir.glob("*.summary.json"))
if not paths:
    raise SystemExit(f"No *.summary.json files found in {args.input_dir}")
rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
if args.max_envs is not None:
    if args.max_envs <= 0:
        raise SystemExit("--max-envs must be positive")
    for row in rows:
        csv_file = args.input_dir / f"{row['model']}.csv"
        with csv_file.open(newline="", encoding="utf-8") as stream:
            samples = [sample for sample in csv.DictReader(stream) if int(sample["env_id"]) < args.max_envs]
        pos = np.asarray([float(sample["final_pos_mean_m"]) for sample in samples])
        ori = np.asarray([float(sample["final_ori_mean_rad"]) for sample in samples])
        success = np.asarray([int(sample["success_5cm_5deg"]) for sample in samples])
        terminated = np.asarray([int(sample["terminated"]) for sample in samples])
        action_rate = np.asarray([float(sample["action_rate_l2_mean"]) for sample in samples])
        action_rms = np.asarray([float(sample["action_rms"]) for sample in samples])
        finite_pos = pos[np.isfinite(pos)]
        finite_ori = ori[np.isfinite(ori)]
        finite_action_rate = action_rate[np.isfinite(action_rate)]
        finite_action_rms = action_rms[np.isfinite(action_rms)]
        abort = np.asarray(["simulator_abort" in sample["termination_reason"] for sample in samples])
        row.update(
            num_commands=len(samples),
            position_mean_m=float(np.mean(finite_pos)),
            position_median_m=float(np.median(finite_pos)),
            position_p95_m=float(np.quantile(finite_pos, 0.95)),
            orientation_mean_rad=float(np.mean(finite_ori)),
            orientation_median_rad=float(np.median(finite_ori)),
            orientation_p95_rad=float(np.quantile(finite_ori, 0.95)),
            success_5cm_5deg_rate=float(np.mean(success)),
            termination_rate=float(np.mean(terminated)),
            rollout_abort_rate=float(np.mean(abort)),
            action_rate_l2_mean=float(np.mean(finite_action_rate)),
            action_rate_l2_p95=float(np.quantile(finite_action_rate, 0.95)),
            action_rms_mean=float(np.mean(finite_action_rms)),
        )
order = {
    "full_baseline": 0,
    "no_latent": 1,
    "dreamwaq_cenet": 2,
    "matched_additive_reward": 3,
    "privileged_oracle": 4,
    "additive_reward": 5,
}
rows.sort(key=lambda row: order.get(row["model"], 99))

suffix = f"_{args.max_envs}" if args.max_envs is not None else ""
csv_path = args.input_dir / f"summary{suffix}.csv"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

md_path = args.input_dir / f"summary{suffix}.md"
columns = (
    "model",
    "num_commands",
    "success_5cm_5deg_rate",
    "position_mean_m",
    "position_median_m",
    "position_p95_m",
    "orientation_mean_rad",
    "orientation_p95_rad",
    "termination_rate",
    "rollout_abort_rate",
    "action_rate_l2_mean",
    "action_rate_l2_p95",
    "action_rms_mean",
    "peak_cuda_memory_mib",
    "wall_time_s",
)
lines = [
    "| " + " | ".join(columns) + " |",
    "| " + " | ".join("---" for _ in columns) + " |",
]
for row in rows:
    values = []
    for column in columns:
        value = row[column]
        values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
    lines.append("| " + " | ".join(values) + " |")
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
