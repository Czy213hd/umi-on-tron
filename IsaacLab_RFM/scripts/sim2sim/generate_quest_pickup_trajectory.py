#!/usr/bin/env python3
"""Generate a smooth Quest-like ground-pickup trajectory for sim2sim.

The trajectory uses the same base-frame command convention as the Quest
teleoperation script.  It deliberately keeps low-amplitude, low-frequency
controller drift in X/Y, roll and yaw, while pitch performs the visible pickup
motion.  A payload mass channel marks the grasp interval for the
figure/simulation harness.
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "data/quest_ground_pickup_to_high.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument(
        "--pickup-z",
        type=float,
        default=-0.22,
        help="Lowest base-frame link6 Z target during pickup (default: -0.22 m).",
    )
    return parser.parse_args()


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def transition(
    time: np.ndarray,
    start_time: float,
    end_time: float,
    start_value: float,
    end_value: float,
) -> np.ndarray:
    blend = smoothstep((time - start_time) / (end_time - start_time))
    return start_value + (end_value - start_value) * blend


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


def generate(duration: float, dt: float, payload_mass: float, pickup_z: float) -> dict:
    if duration < 9.0:
        raise ValueError("duration must be at least 9 seconds to preserve all pickup phases")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if payload_mass < 0.0:
        raise ValueError("payload mass must be non-negative")
    if not math.isfinite(pickup_z):
        raise ValueError("pickup Z must be finite")

    time = np.arange(0.0, duration, dt, dtype=np.float64)

    # Base-frame link6 target.  The Quest program starts from the measured
    # reset pose and adds +0.15 m in base X, which is approximately this pose.
    # The command then descends near the floor, pauses to grasp, and raises the
    # payload to the high point.  Once grasped, X advances 0.20 m in the base
    # forward direction; Y contains no deliberate translation beyond
    # controller-like hand jitter.
    x = np.full_like(time, 0.276)
    x = np.where(time >= 5.0, transition(time, 5.0, 7.0, 0.276, 0.476), x)
    x = np.where(time >= 7.0, 0.476, x)
    z = np.full_like(time, 0.252)
    z = np.where(time >= 1.5, transition(time, 1.5, 4.0, 0.252, pickup_z), z)
    z = np.where(time >= 4.0, pickup_z, z)
    z = np.where(time >= 5.0, transition(time, 5.0, 8.5, pickup_z, 0.70), z)
    z = np.where(time >= 8.5, 0.70, z)

    # Quest orientation is a delta about the pose captured on Grip press.  The
    # reset link6 pitch is approximately 0.435 rad (25 deg), so keep that as
    # the baseline instead of commanding world-frame identity.
    pitch = np.full_like(time, 0.435)
    pitch = np.where(time >= 1.5, transition(time, 1.5, 4.0, 0.435, 0.70), pitch)
    pitch = np.where(time >= 4.0, 0.70, pitch)
    pitch = np.where(time >= 5.0, transition(time, 5.0, 8.5, 0.70, 0.435), pitch)
    pitch = np.where(time >= 8.5, 0.435, pitch)

    # Smooth controller-like drift: millimetres in position and about one degree
    # in roll/yaw.  Frequencies are deliberately below the 50 Hz policy rate.
    x += 0.0030 * np.sin(2.0 * np.pi * 0.63 * time + 0.4)
    x += 0.0012 * np.sin(2.0 * np.pi * 1.41 * time + 1.2)
    y = 0.0040 * np.sin(2.0 * np.pi * 0.47 * time + 0.7)
    y += 0.0018 * np.sin(2.0 * np.pi * 1.17 * time + 2.0)
    z += 0.0020 * np.sin(2.0 * np.pi * 0.56 * time + 0.2)

    roll = math.radians(0.8) * np.sin(2.0 * np.pi * 0.39 * time + 0.8)
    roll += math.radians(0.25) * np.sin(2.0 * np.pi * 1.31 * time)
    yaw = math.radians(1.0) * np.sin(2.0 * np.pi * 0.31 * time + 1.6)
    yaw += math.radians(0.3) * np.sin(2.0 * np.pi * 1.09 * time + 0.5)
    pitch += math.radians(0.45) * np.sin(2.0 * np.pi * 0.52 * time + 0.3)

    positions = np.column_stack((x, y, z))
    rotations = [rotation_from_rpy(r, p, y_) for r, p, y_ in zip(roll, pitch, yaw)]
    axis_angles = np.stack([axis_angle_from_rotation(rotation) for rotation in rotations])

    gripper_width = np.full_like(time, 0.085)
    gripper_width = np.where(
        time >= 4.25,
        transition(time, 4.25, 4.75, 0.085, 0.018),
        gripper_width,
    )
    payload = np.zeros_like(time)
    payload = np.where(
        time >= 4.55,
        transition(time, 4.55, 4.85, 0.0, payload_mass),
        payload,
    )

    return {
        "t": time,
        "ee_pos": positions,
        "ee_axis_angle": axis_angles,
        "gripper_width": gripper_width,
        "payload_mass": payload,
        "metadata": {
            "position_mode": "absolute_base",
            "trajectory_name": "quest_ground_pickup_to_high",
            "description": (
                "Quest-like ground pickup: approach, pitch/down, grasp, "
                "unpitch/lift, advance 0.20 m in base X, and high hold"
            ),
            "object_position_world": [0.39, 0.0, 0.09],
            "grasp_time_s": 4.55,
            "pickup_z_base_m": pickup_z,
            "pickup_tip_offset_link6": [0.15, 0.0, 0.0],
            "phase_times_s": {
                "approach_end": 1.5,
                "ground_reached": 4.0,
                "grasp": 4.55,
                "lift_start": 5.0,
                "forward_complete": 7.0,
                "highest_point": 8.5,
                "end": duration,
            },
        },
    }


def main() -> int:
    args = parse_args()
    episode = generate(args.duration, args.dt, args.payload_mass, args.pickup_z)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump([episode], file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[trajectory] wrote {args.output.expanduser().resolve()}")
    print(
        f"[trajectory] frames={len(episode['t'])}, duration={args.duration:g}s, "
        f"payload={args.payload_mass:g}kg"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
