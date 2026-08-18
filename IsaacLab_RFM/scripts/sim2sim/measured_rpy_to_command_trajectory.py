#!/usr/bin/env python3
"""Build a replay trajectory using a prior sim2sim measured RPY as command."""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "IsaacLab_RFM/outputs/sim2sim_figures/keyboard_pickup_run_01_tracking.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "data/keyboard_pickup_run_01_measured_rpy_command.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def axis_angle_from_rotation(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    if math.pi - angle < 1.0e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    with np.load(input_path) as data:
        tracking_frame = str(data["tracking_frame"])
        time = np.asarray(data["time"], dtype=np.float64)
        target_position = np.asarray(data["target_position"], dtype=np.float64)
        measured_rpy = np.asarray(data["measured_rpy_world"], dtype=np.float64)

    if tracking_frame != "base":
        raise ValueError(
            f"Expected a base-frame tracking recording, received {tracking_frame!r}"
        )
    if time.ndim != 1 or target_position.shape != (len(time), 3):
        raise ValueError("Unexpected time/target_position shape")
    if measured_rpy.shape != target_position.shape:
        raise ValueError("Measured RPY shape does not match target position")
    if not all(np.isfinite(value).all() for value in (time, target_position, measured_rpy)):
        raise ValueError("Input recording contains non-finite values")

    rotations = np.stack([rotation_from_rpy(*rpy) for rpy in measured_rpy])
    axis_angles = np.stack(
        [axis_angle_from_rotation(rotation) for rotation in rotations]
    )
    relative_time = time - time[0]
    episode = {
        "t": relative_time,
        "ee_pos": target_position,
        "ee_axis_angle": axis_angles,
        "gripper_width": np.zeros(len(time), dtype=np.float64),
        "payload_mass": np.zeros(len(time), dtype=np.float64),
        "metadata": {
            "position_mode": "absolute_base",
            "trajectory_name": "keyboard_pickup_run_01_measured_rpy_command",
            "description": (
                "Second-pass trajectory: prior target XYZ with prior measured base-frame "
                "RPY used as the new orientation command"
            ),
            "source_tracking_npz": str(input_path),
            "orientation_source": "prior_measured_rpy_base",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump([episode], file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[measured-rpy-command] output: {args.output.expanduser().resolve()}")
    print(
        f"[measured-rpy-command] samples={len(time)}, "
        f"duration={relative_time[-1]:.2f}s, frame=base"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
