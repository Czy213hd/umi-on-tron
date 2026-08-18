#!/usr/bin/env python3
"""Plot raw and smoothed XYZ commands from a keyboard teleoperation recording."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sim2sim-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_INPUT = REPO_ROOT / "data/keyboard_pickup_run_01.npz"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "IsaacLab_RFM/outputs/sim2sim_figures/keyboard_pickup_run_01_xyz_command.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_OUTPUT.with_suffix(".pdf"))
    parser.add_argument(
        "--trim-padding",
        type=float,
        default=1.0,
        help="Seconds retained before and after the active keyboard interval (default: 1).",
    )
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def active_bounds(time: np.ndarray, command: np.ndarray, padding: float) -> tuple[float, float]:
    change = np.linalg.norm(np.diff(command, axis=0), axis=1)
    changed = np.flatnonzero(change > 1.0e-8)
    if len(changed) == 0:
        return float(time[0]), float(time[-1])
    start = max(float(time[0]), float(time[changed[0]]) - padding)
    end = min(float(time[-1]), float(time[changed[-1] + 1]) + padding)
    return start, end


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.trim_padding) or args.trim_padding < 0.0:
        raise ValueError("--trim-padding must be finite and non-negative")

    input_path = args.input.expanduser().resolve()
    with np.load(input_path) as data:
        time = np.asarray(data["time_from_start"], dtype=np.float64)
        raw = np.asarray(data["target_position_base"], dtype=np.float64)
        smooth = np.asarray(data["target_position_base_smooth"], dtype=np.float64)
        sample_rate = float(data["sample_rate_hz"])
        smoothing_sigma = float(data["smoothing_sigma_s"])

    if time.ndim != 1 or raw.shape != (len(time), 3) or smooth.shape != raw.shape:
        raise ValueError("Recording does not contain compatible XYZ command arrays")
    if not all(np.isfinite(value).all() for value in (time, raw, smooth)):
        raise ValueError("Recording contains non-finite command values")

    if args.no_trim:
        start, end = float(time[0]), float(time[-1])
    else:
        start, end = active_bounds(time, raw, args.trim_padding)

    colors = ("#2455ff", "#ef3b2c", "#2c8d35")
    names = ("X", "Y", "Z")
    figure, axis = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    for index, (name, color) in enumerate(zip(names, colors)):
        axis.step(
            time,
            raw[:, index],
            where="post",
            color=color,
            linewidth=0.85,
            linestyle=":",
            alpha=0.62,
            label=f"{name} raw keyboard",
        )
        axis.plot(
            time,
            smooth[:, index],
            color=color,
            linewidth=2.0,
            label=f"{name} smoothed command",
        )

    axis.set_xlim(start, end)
    visible = (time >= start) & (time <= end)
    visible_values = np.concatenate((raw[visible].ravel(), smooth[visible].ravel()))
    value_span = max(float(np.ptp(visible_values)), 0.1)
    axis.set_ylim(
        float(np.min(visible_values)) - 0.08 * value_span,
        float(np.max(visible_values)) + 0.12 * value_span,
    )
    axis.set_xlabel("Recording time [s]", fontsize=11)
    axis.set_ylabel("Base-frame XYZ command [m]", fontsize=11)
    figure.suptitle(
        "Keyboard teleoperation XYZ command",
        fontsize=15,
        fontweight="semibold",
    )
    axis.set_title(
        f"50 Hz recording · Gaussian smoothing σ={smoothing_sigma:.2f} s · "
        f"active view {start:.2f}–{end:.2f} s",
        fontsize=9,
        color="#555d6b",
        pad=9,
    )
    axis.grid(True, color="#d9dee8", linewidth=0.75, alpha=0.8)
    axis.axhline(0.0, color="#7f8795", linewidth=0.7, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    ranges = np.ptp(raw, axis=0)
    axis.text(
        0.985,
        0.965,
        f"Raw command range\nX {ranges[0]:.3f} m · Y {ranges[1]:.3f} m · Z {ranges[2]:.3f} m",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c8ced9"},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor="white")
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.pdf, facecolor="white")
    plt.close(figure)
    print(f"[xyz-command] PNG: {args.output.expanduser().resolve()}")
    print(f"[xyz-command] PDF: {args.pdf.expanduser().resolve()}")
    print(
        f"[xyz-command] samples={len(time)}, duration={time[-1]:.2f}s, "
        f"view={start:.2f}..{end:.2f}s, sample_rate={sample_rate:g}Hz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
