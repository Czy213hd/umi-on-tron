#!/usr/bin/env python3
"""Run the SF_TRON1A + ARXR5 arm policy in MuJoCo.

The inference chain matches the IsaacLab/RSL-RL export:

    10 x 57 contact observations -> contactNet -> GRU -> 67-D latent
    67-D policy observation + 67-D latent -> actor -> 14 joint targets

If ``--command`` is omitted, the program asks for a final end-effector pose
as ``x y z roll pitch yaw`` before opening the viewer.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import select
import sys
import termios
import threading
import time
import tty
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


SCRIPT_PATH = Path(__file__).resolve()
ISAACLAB_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = ISAACLAB_ROOT.parent

DEFAULT_MJCF = (
    ISAACLAB_ROOT
    / "source/ext_loco/ext_loco/assets/SF_TRON1A_ARXR5ARM/assembly.xml"
)
DEPLOYED_MODEL_DIR = (
    REPO_ROOT
    / "tron1_ws/src/tron1-rl-deploy-arm/src/robot_controllers/config/"
    "pointfoot/SF_TRON1A_ARX5ARM/policy"
)
WBC_LOG_ROOT = Path(os.environ.get("WBC_LOG_ROOT", "/media/edwin/ChenJing26/WBC_logs"))
DEFAULT_TRAJECTORY = Path("/home/phi5090ii/UMI-ON-TRON/data/pushing.pkl")
# The checkpoint tracks J6's child rigid body, link6. No tip offset is applied.
TIP_OFFSET_POS = np.zeros(3, dtype=np.float64)
TIP_OFFSET_RPY = (0.0, 0.0, 0.0)

# This is the Isaac Lab articulation/action order measured at runtime for the
# training asset.  MuJoCo state and actuator addresses are explicitly gathered
# by name below, so they must be exposed to the policy in this exact order.
JOINT_NAMES = (
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
    "J1",
    "ankle_L_Joint",
    "ankle_R_Joint",
    "J2",
    "J3",
    "J4",
    "J5",
    "J6",
)
LEG_NAMES = (
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "ankle_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
    "ankle_R_Joint",
)
ARM_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
LEG_IDS = np.array([JOINT_NAMES.index(name) for name in LEG_NAMES], dtype=int)
ARM_IDS = np.array([JOINT_NAMES.index(name) for name in ARM_NAMES], dtype=int)
J6_ID = JOINT_NAMES.index("J6")
# JointPositionAction scales saved in this checkpoint's params/env.yaml:
# legs/ankles=0.6, J1-J3=0.3, J4-J6=0.2.
ACTION_SCALE = np.array(
    [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.6, 0.6, 0.3, 0.3, 0.2, 0.2, 0.2],
    dtype=np.float64,
)
DEFAULT_JOINT_POS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
# IsaacLab clamps the reset state to the 0.9 soft joint-position limits.
# J3's hard range is [-0.1, 3.2], so its reset value becomes 0.065 rad even
# though the configured default is zero.  Use the state measured after reset,
# because this is what the first policy/contactNet observation actually sees.
ISAAC_RESET_JOINT_POS = DEFAULT_JOINT_POS.copy()
ISAAC_RESET_JOINT_POS[JOINT_NAMES.index("J3")] = 0.065

# IsaacLab actuator gains used by LIMX_SF_TRON1A_ARM.
KP = np.array(
    [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 18.0, 45.0, 45.0, 18.0, 18.0, 4.0, 4.0, 4.0],
    dtype=np.float64,
)
KD = np.array(
    [1.8, 1.8, 1.8, 1.8, 1.8, 1.8, 1.0, 0.8, 0.8, 1.0, 1.0, 0.5, 0.5, 0.5],
    dtype=np.float64,
)
TORQUE_LIMIT = np.array(
    [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 18.0, 40.0, 40.0, 18.0, 18.0, 3.0, 3.0, 3.0],
    dtype=np.float64,
)
VELOCITY_LIMIT = np.array(
    [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 3.14, 40.0, 40.0, 3.14, 3.14, 3.9, 3.9, 3.9],
    dtype=np.float64,
)

# The policy clock must match training exactly (50 Hz). MuJoCo defaults to a
# finer 1 ms contact substep because its mesh contact becomes unstable at the
# PhysX training step of 5 ms. Use --physics-dt 0.005 for strict clock tests.
TRAINING_PHYSICS_DT = 0.005
DEFAULT_PHYSICS_DT = 0.001
POLICY_DT = 0.02
HISTORY_LENGTH = 10
OBS_DIM = 67
CONTACT_OBS_DIM = 57
ACTION_DIM = 14
# A single MuJoCo sliding coefficient cannot represent PhysX's independent
# static [0.7, 1.3] and dynamic [0.6, 1.2] randomization. 0.9 lies in both
# training ranges and is therefore the deterministic parity value.
MUJOCO_CONTACT_FRICTION = "0.9 0.005 0.0001"


def format_named(names: tuple[str, ...], values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return " ".join(
        f"{name}={values[index]: .4f}"
        for index, name in enumerate(names[: values.size])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo sim2sim for the SF_TRON1A ARXR5Arm three-ONNX policy."
    )
    parser.add_argument(
        "--command",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="Final EE pose. Position is metres and RPY is radians.",
    )
    parser.add_argument(
        "--command-frame",
        choices=("world", "base"),
        default="world",
        help="Frame of --command (default: world).",
    )
    parser.add_argument(
        "--eef-x-offset",
        type=float,
        default=None,
        help=(
            "Override --command with a fixed world-frame target whose position is the "
            "initial gripper-base/camera-center position plus this many metres along world +X; preserve "
            "the initial EEF orientation."
        ),
    )
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing actor.onnx, contactNet.onnx and gru.onnx. "
        "Default: newest exported run under WBC_LOG_ROOT, then the legacy local log.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration in seconds; 0 runs until the viewer closes. "
        "Default: the 10 s training episode manually, or one complete trajectory.",
    )
    parser.add_argument(
        "--freeze-after",
        type=float,
        default=None,
        help=(
            "Stop advancing physics after this many simulation seconds while keeping "
            "the viewer open on the final pose. Use --duration 0 for an indefinite hold."
        ),
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=0.8,
        help="Initial base_Link height in metres (default: 0.8, matching IsaacLab play).",
    )
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=DEFAULT_PHYSICS_DT,
        help="MuJoCo substep in seconds (default: 0.001 for stable mesh contact; "
        "use 0.005 to match the IsaacLab physics clock exactly).",
    )
    parser.add_argument(
        "--se3-decrease-vel",
        type=float,
        default=1.0,
        help="SE(3) reference decay rate. Training samples [0.5, 1.4] per command; "
        "1.0 is the deterministic training-domain value, while 0 matches fixed Command-Play.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=None,
        help=(
            "Override the JointPositionAction scale uniformly for all 14 joints. "
            "Use 1.0 for legacy deploy2; omit for the newer grouped scales."
        ),
    )
    latent_group = parser.add_mutually_exclusive_group()
    latent_group.add_argument(
        "--sample-latent",
        dest="sample_latent",
        action="store_true",
        help="Sample the predicted latent distribution (default; matches training/Isaac play).",
    )
    latent_group.add_argument(
        "--mean-latent",
        dest="sample_latent",
        action="store_false",
        help="Use the latent mean for deterministic diagnostics (not training parity).",
    )
    parser.set_defaults(sample_latent=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--render-fps",
        type=float,
        default=60.0,
        help="Viewer synchronization rate; independent of the 1 kHz physics rate.",
    )
    parser.add_argument(
        "--free-camera",
        action="store_true",
        help="Use a stationary free camera instead of following base_Link.",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=135.0,
        help="Initial MuJoCo camera azimuth in degrees (150 is the robot's 1 o'clock direction).",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=-18.0,
        help="Initial MuJoCo camera elevation in degrees (default: -18).",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.4,
        help="Initial MuJoCo camera distance in metres (default: 2.4).",
    )
    parser.add_argument(
        "--hide-current-ee-frame",
        action="store_true",
        help="Hide the current link6 RGB frame and the built-in red EEF marker.",
    )
    parser.add_argument(
        "--door",
        action="store_true",
        help="Add the deployment door scene. Disabled by default to match the flat training scene.",
    )
    parser.add_argument(
        "--keyboard-step",
        "--keyboard",
        dest="keyboard_step",
        type=float,
        default=0.02,
        help="Target-position increment per key press in metres (default: 0.02).",
    )
    parser.add_argument(
        "--keyboard-rotation-step",
        "--keyboard-rpy-step",
        dest="keyboard_rotation_step",
        type=float,
        default=math.radians(5.0),
        help="Target roll/pitch/yaw increment per key press in radians (default: 5 degrees).",
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        help=(
            "Record keyboard teleoperation at the 50 Hz policy rate to an NPZ file. "
            "Raw and smoothed replayable PKL trajectories are written beside it."
        ),
    )
    parser.add_argument(
        "--record-smoothing",
        type=float,
        default=0.12,
        help=(
            "Gaussian smoothing sigma in seconds for the exported keyboard trajectory "
            "(default: 0.12; 0 disables smoothing)."
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help=f"Play a pushing.pkl trajectory (known file: {DEFAULT_TRAJECTORY}).",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
        help="Episode index inside the pickle file (default: 0).",
    )
    parser.add_argument(
        "--trajectory-start-delay",
        type=float,
        default=3.0,
        help="Seconds to stabilize before anchoring and playing the trajectory.",
    )
    parser.add_argument(
        "--trajectory-loop",
        action="store_true",
        help="Loop the selected episode and re-anchor each repetition.",
    )
    parser.add_argument(
        "--trajectory-speed",
        type=float,
        default=1.0,
        help="Trajectory playback speed multiplier (default: 1.0; use 0.25 for quarter speed).",
    )
    parser.add_argument(
        "--no-planar-center",
        action="store_true",
        help="Disable the XY centering used by IsaacLab PicklePoseSequenceCommand.",
    )
    parser.add_argument(
        "--pickup-object",
        action="store_true",
        help=(
            "Show the trajectory metadata object_position_world as a visual pickup object; "
            "after payload_mass becomes positive it follows the measured gripper point."
        ),
    )
    parser.add_argument(
        "--payload-gravity",
        action="store_true",
        help=(
            "Apply each trajectory frame's payload_mass as downward gravity at the gripper point."
        ),
    )
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument(
        "--leg-debug",
        action="store_true",
        help="Print leg q, raw/effective actions, desired/cmd positions, command step, and tracking error.",
    )
    parser.add_argument(
        "--arm-max-step",
        type=float,
        default=0.0,
        help="Maximum arm target-position change per 50 Hz policy update in radians; 0 disables it.",
    )
    parser.add_argument(
        "--max-leg-step",
        type=float,
        default=0.0,
        help="Maximum leg target-position change per 50 Hz policy update in radians; 0 disables it.",
    )
    parser.add_argument(
        "--lock-j6",
        action="store_true",
        help="Force J6 to remain at --lock-j6-angle and expose the locked command as last_action.",
    )
    parser.add_argument(
        "--lock-j6-angle",
        type=float,
        default=0.0,
        help="Fixed J6 position in radians when --lock-j6 is enabled (default: 0).",
    )
    return parser.parse_args()


def newest_exported_model_dir() -> Path:
    # Training now writes to WBC_LOG_ROOT. Keep the in-repository location as
    # a fallback so old local runs remain usable.
    log_roots = (
        WBC_LOG_ROOT,
        ISAACLAB_ROOT / "logs/rsl_rl/ImplicitOneStageARXR5Arm",
    )
    candidates = [
        path
        for log_root in log_roots
        for path in log_root.glob("*/exported")
        if all((path / name).is_file() for name in ("actor.onnx", "contactNet.onnx", "gru.onnx"))
    ]
    if candidates:
        return max(candidates, key=lambda path: (path.parent.stat().st_mtime_ns, path.parent.name))
    return DEPLOYED_MODEL_DIR


def read_command(command: list[float] | None) -> np.ndarray:
    if command is not None:
        return np.asarray(command, dtype=np.float64)
    default = "0.15 0.0 1.0 0.0 0.0 0.0"
    print("输入末端最终点：x y z roll pitch yaw")
    print(f"单位：位置 m，姿态 rad。直接回车使用默认值：{default}")
    while True:
        text = input("> ").strip() or default
        try:
            values = np.asarray([float(item) for item in text.split()], dtype=np.float64)
        except ValueError:
            print("输入包含非数字，请重新输入。")
            continue
        if values.shape == (6,) and np.isfinite(values).all():
            return values
        print("必须输入 6 个有限数值。")


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


def rpy_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Return ZYX roll, pitch and yaw angles from a rotation matrix."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if horizontal > 1.0e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(-float(rotation[2, 0]), horizontal)
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        pitch = math.atan2(-float(rotation[2, 0]), horizontal)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(float(cosine)))


def axis_angle_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to an axis-angle vector."""
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


def gaussian_smooth(values: np.ndarray, sigma_seconds: float, sample_dt: float) -> np.ndarray:
    """Smooth a sampled vector signal with an edge-preserving Gaussian kernel."""
    values = np.asarray(values, dtype=np.float64)
    if sigma_seconds <= 0.0 or len(values) < 2:
        return values.copy()
    sigma_samples = sigma_seconds / sample_dt
    radius = max(1, int(math.ceil(3.0 * sigma_samples)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_samples))
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.column_stack(
        [np.convolve(padded[:, index], kernel, mode="valid") for index in range(values.shape[1])]
    )


def rotation_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1.0e-10:
        return np.eye(3)
    x, y, z = axis_angle / angle
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


class PickleTrajectory:
    """Playback compatible with IsaacLab's PicklePoseSequenceCommand."""

    def __init__(
        self,
        path: Path,
        episode_index: int,
        start_delay: float,
        loop: bool,
        planar_center: bool,
        playback_speed: float,
    ):
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory not found: {path}")
        with path.open("rb") as file:
            episodes = pickle.load(file)
        if isinstance(episodes, dict) and "episodes" in episodes:
            episodes = episodes["episodes"]
        if not isinstance(episodes, (list, tuple)) or not episodes:
            raise ValueError("Trajectory pickle must contain a non-empty episode list")
        if not -len(episodes) <= episode_index < len(episodes):
            raise IndexError(f"trajectory index {episode_index} outside [0, {len(episodes) - 1}]")

        self.path = path
        self.episode_index = episode_index % len(episodes)
        episode = episodes[self.episode_index]
        metadata = episode.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Trajectory metadata must be a dictionary when provided")
        self.metadata = metadata
        self.position_mode = metadata.get("position_mode", "relative_to_playback_start")
        if self.position_mode not in (
            "relative_to_playback_start",
            "relative_xy_absolute_world_z",
            "absolute_world",
            "absolute_base",
        ):
            raise ValueError(f"Unsupported trajectory position_mode: {self.position_mode}")
        self.positions = np.asarray(episode["ee_pos"], dtype=np.float64).copy()
        axis_angles = np.asarray(episode["ee_axis_angle"], dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"Unexpected ee_pos shape: {self.positions.shape}")
        if axis_angles.shape != self.positions.shape:
            raise ValueError(
                f"ee_axis_angle shape {axis_angles.shape} does not match ee_pos {self.positions.shape}"
            )
        if planar_center and self.position_mode not in ("absolute_world", "absolute_base"):
            if len(self.positions) < 4:
                raise ValueError("planar_center requires at least four trajectory frames")
            self.positions[:, :2] -= self.positions[1:4, :2].mean(axis=0)

        self.rotations = np.stack([rotation_from_axis_angle(value) for value in axis_angles])
        self.payload_mass = np.asarray(
            episode.get("payload_mass", np.zeros(len(self.positions))), dtype=np.float64
        )
        if self.payload_mass.shape != (len(self.positions),):
            raise ValueError("trajectory payload_mass must have one value per pose frame")
        if not np.isfinite(self.payload_mass).all() or np.any(self.payload_mass < 0.0):
            raise ValueError("trajectory payload_mass must contain finite non-negative values")
        self.object_position_world = np.asarray(
            metadata.get("object_position_world", [0.64, 0.0, 0.06]), dtype=np.float64
        )
        self.pickup_tip_offset = np.asarray(
            metadata.get("pickup_tip_offset_link6", [0.15, 0.0, 0.0]),
            dtype=np.float64,
        )
        self.grasp_time = float(metadata.get("grasp_time_s", math.inf))
        if self.object_position_world.shape != (3,) or self.pickup_tip_offset.shape != (3,):
            raise ValueError("pickup object position and tip offset metadata must be 3-vectors")
        if math.isnan(self.grasp_time) or self.grasp_time < 0.0:
            raise ValueError("trajectory grasp_time_s must be non-negative")
        time_samples = np.asarray(episode.get("t", []), dtype=np.float64)
        if len(time_samples) >= 2:
            self.sample_dt = float(np.median(np.diff(time_samples)))
        else:
            self.sample_dt = 0.005
        if self.sample_dt <= 0:
            raise ValueError(f"Invalid trajectory sample dt: {self.sample_dt}")
        if not math.isfinite(playback_speed) or playback_speed <= 0:
            raise ValueError("--trajectory-speed must be a finite number greater than zero")

        tip_rotation = rotation_from_rpy(*TIP_OFFSET_RPY)
        self.tip_rotation_inverse = tip_rotation.T
        self.tip_position_inverse = -self.tip_rotation_inverse @ TIP_OFFSET_POS
        self.start_delay = max(0.0, float(start_delay))
        self.loop = loop
        self.playback_speed = float(playback_speed)
        self.source_duration = len(self.positions) * self.sample_dt
        self.duration = self.source_duration / self.playback_speed
        self.world_offset = np.zeros(3, dtype=np.float64)
        self.command_origin: np.ndarray | None = None
        self.current_cycle = -1
        self.finished_message_printed = False
        self.current_frame = 0

    def translate_offset(self, delta: np.ndarray) -> None:
        self.world_offset += np.asarray(delta, dtype=np.float64)

    def update(self, simulation: "Sim2Sim") -> None:
        elapsed = simulation.data.time - self.start_delay
        if elapsed < 0:
            return

        if self.loop:
            cycle = int(elapsed // self.duration)
            playback_time = elapsed - cycle * self.duration
        else:
            cycle = 0
            playback_time = min(elapsed, self.duration)

        new_cycle = self.command_origin is None or cycle != self.current_cycle
        if new_cycle:
            self.command_origin = simulation.data.xpos[simulation.ee_body_id].copy()
            self.current_cycle = cycle
            self.finished_message_printed = False
            print(
                f"[trajectory] episode={self.episode_index}, cycle={cycle}, "
                f"origin={self.command_origin.round(4).tolist()}"
            )

        source_time = min(
            playback_time * self.playback_speed,
            self.source_duration - self.sample_dt,
        )
        frame = min(int(source_time / self.sample_dt), len(self.positions) - 1)
        self.current_frame = frame
        tip_position = self.positions[frame]
        tip_rotation = self.rotations[frame]
        link_position = tip_position + tip_rotation @ self.tip_position_inverse
        link_rotation = tip_rotation @ self.tip_rotation_inverse
        if self.position_mode == "absolute_base":
            simulation.command_frame = "base"
            target_position = link_position + self.world_offset
        elif self.position_mode == "absolute_world":
            simulation.command_frame = "world"
            target_position = link_position + self.world_offset
        else:
            simulation.command_frame = "world"
            target_position = self.command_origin + link_position + self.world_offset
        if self.position_mode == "relative_xy_absolute_world_z":
            # Preserve the normal relative anchoring in the horizontal plane,
            # but interpret the PKL Z value as an exact world-frame height.
            target_position[2] = link_position[2] + self.world_offset[2]
        simulation.set_target(target_position, link_rotation, reset_reference=new_cycle)

        if not self.loop and elapsed >= self.duration and not self.finished_message_printed:
            print("[trajectory] playback complete; holding the final pose.")
            self.finished_message_printed = True


def load_sim_model(
    mjcf_path: Path,
    *,
    physics_dt: float = DEFAULT_PHYSICS_DT,
    add_door: bool = False,
    hide_current_ee_frame: bool = False,
    add_pickup_object: bool = False,
) -> mujoco.MjModel:
    """Add a floor and target marker without changing the robot MJCF on disk."""
    mjcf_path = mjcf_path.expanduser().resolve()
    if not mjcf_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = compiler.get("meshdir", ".")
    compiler.set("meshdir", str((mjcf_path.parent / meshdir).resolve()))

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(physics_dt))
    if math.isclose(physics_dt, TRAINING_PHYSICS_DT):
        option.set("integrator", "implicitfast")
        option.set("solver", "Newton")
        option.set("iterations", "100")
        option.set("ls_iterations", "20")
        option.set("noslip_iterations", "10")

    # The MuJoCo defaults (solref=0.02 1) are visibly too soft for these
    # high-resolution foot collision meshes. Keep the contact stable and firm
    # without making it perfectly rigid.
    collision_default = root.find("./default/default[@class='collision']/geom")
    if collision_default is not None:
        collision_default.set("friction", MUJOCO_CONTACT_FRICTION)
        collision_default.set("solref", "0.005 1")
        collision_default.set("solimp", "0.95 0.99 0.001")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(
        {
            "diffuse": "0.34 0.35 0.37",
            "ambient": "0.10 0.11 0.12",
            "specular": "0.035 0.035 0.035",
        }
    )

    # Procedural studio sky and checkerboard floor. They live only in the
    # in-memory model, so the source robot MJCF remains reusable by deployments
    # that do not want this viewer styling.
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody_index = next(
            (index for index, element in enumerate(root) if element.tag == "worldbody"),
            len(root),
        )
        root.insert(worldbody_index, asset)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "sim2sim_studio_skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.48 0.60 0.72",
            "rgb2": "0.92 0.94 0.96",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "sim2sim_checker_texture",
            "type": "2d",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.26 0.31 0.37",
            "rgb2": "0.42 0.48 0.55",
            "markrgb": "0.58 0.64 0.70",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "sim2sim_checker_material",
            "texture": "sim2sim_checker_texture",
            "texrepeat": "8 8",
            "texuniform": "true",
            "reflectance": "0.08",
        },
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            {
                "name": "sim2sim_floor",
                "type": "plane",
                "size": "0 0 0.1",
                "material": "sim2sim_checker_material",
                "friction": MUJOCO_CONTACT_FRICTION,
                "condim": "3",
                "solref": "0.005 1",
                "solimp": "0.95 0.99 0.001",
            },
        ),
    )
    worldbody.insert(
        1,
        ET.Element(
            "light",
            {
                "name": "sim2sim_key_light",
                "pos": "-2 -2 4",
                "dir": "0.4 0.3 -1",
                "directional": "true",
                "diffuse": "0.38 0.40 0.44",
                "ambient": "0.025 0.028 0.032",
                "specular": "0.07 0.07 0.07",
                "castshadow": "true",
            },
        ),
    )
    worldbody.insert(
        2,
        ET.Element(
            "light",
            {
                "name": "sim2sim_fill_light",
                "pos": "2 2 3",
                "dir": "-0.5 -0.3 -1",
                "directional": "true",
                "diffuse": "0.10 0.12 0.15",
                "ambient": "0.025 0.028 0.032",
                "specular": "0.01 0.01 0.01",
                "castshadow": "false",
            },
        ),
    )

    # A fixed doorway with a dynamic door panel. The robot starts at the
    # origin facing +X, so the door plane is placed in front of it at x=0.75 m.
    contact_parameters = {
        "friction": MUJOCO_CONTACT_FRICTION,
        "condim": "3",
        "solref": "0.005 1",
        "solimp": "0.95 0.99 0.001",
    }
    for name, pos, size in (
        ("door_frame_left", "0.75 0.93 0.73", "0.06 0.45 0.73"),
        ("door_frame_right", "0.75 -0.93 0.73", "0.06 0.45 0.73"),
        ("door_frame_header", "0.75 0 1.56", "0.06 0.48 0.10"),
    ):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": pos,
                "size": size,
                "rgba": "0.45 0.46 0.46 1",
                **contact_parameters,
            },
        )

    door = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "sim2sim_door",
            "pos": "0.75 0.45 0.02",
        },
    )
    ET.SubElement(
        door,
        "inertial",
        {
            "pos": "0 -0.445 0.70",
            "mass": "8.25",
            "diaginertia": "1.78 1.28 0.53",
        },
    )
    ET.SubElement(
        door,
        "joint",
        {
            "name": "sim2sim_door_hinge",
            "type": "hinge",
            "axis": "0 0 1",
            "limited": "true",
            "range": "-1.75 1.75",
            "damping": "1.0",
            "frictionloss": "0.2",
            "armature": "0.01",
        },
    )
    ET.SubElement(
        door,
        "geom",
        {
            "name": "sim2sim_door_panel",
            "type": "box",
            "pos": "0 -0.44 0.69",
            "size": "0.025 0.44 0.69",
            "rgba": "0.45 0.18 0.055 1",
            **contact_parameters,
        },
    )

    if not add_door:
        training_extras = {
            "door_frame_left",
            "door_frame_right",
            "door_frame_header",
            "sim2sim_door",
        }
        for element in list(worldbody):
            if element.get("name") in training_extras:
                worldbody.remove(element)
    ET.SubElement(
        door,
        "geom",
        {
            "name": "sim2sim_door_handle_shaft",
            "type": "cylinder",
            "pos": "-0.075 -0.75 1.02",
            "quat": "0.7071068 0 0.7071068 0",
            "size": "0.022 0.05",
            "rgba": "0.12 0.12 0.12 1",
            **contact_parameters,
        },
    )
    ET.SubElement(
        door,
        "geom",
        {
            "name": "sim2sim_door_handle_knob",
            "type": "sphere",
            "pos": "-0.135 -0.75 1.02",
            "size": "0.032",
            "rgba": "0.12 0.12 0.12 1",
            **contact_parameters,
        },
    )

    # A collision-free mocap body renders a slender target frame. Local
    # +X/+Y/+Z are red/green/blue respectively.
    target_frame = ET.Element(
        "body",
        {
            "name": "command_target_frame",
            "mocap": "true",
            "pos": "0.15 0 1",
        },
    )
    ET.SubElement(
        target_frame,
        "site",
        {
            "name": "command_target",
            "type": "sphere",
            "size": "0.006",
            "rgba": "1 1 1 0.95",
            "group": "0",
        },
    )
    for name, endpoint, color in (
        ("command_target_x", "0.11 0 0", "0.95 0.20 0.16 0.95"),
        ("command_target_y", "0 0.11 0", "0.10 0.72 0.34 0.95"),
        ("command_target_z", "0 0 0.11", "0.12 0.38 0.92 0.95"),
    ):
        ET.SubElement(
            target_frame,
            "site",
            {
                "name": name,
                "type": "capsule",
                "fromto": f"0 0 0 {endpoint}",
                "size": "0.0035",
                "rgba": color,
                "group": "0",
            },
        )
        ET.SubElement(
            target_frame,
            "site",
            {
                "name": f"{name}_tip",
                "type": "sphere",
                "pos": endpoint,
                "size": "0.0065",
                "rgba": color,
                "group": "0",
            },
        )
    worldbody.insert(3, target_frame)

    # Optional collision-free mocap payload used by the paper-figure harness.
    # Its gravity load is applied separately at the measured gripper point, so
    # this body is visual-only and cannot destabilize contact at pickup time.
    if add_pickup_object:
        pickup_object = ET.Element(
            "body",
            {
                "name": "sim2sim_pickup_object",
                "mocap": "true",
                "pos": "0.39 0 0.09",
            },
        )
        ET.SubElement(
            pickup_object,
            "geom",
            {
                "name": "sim2sim_pickup_object_geom",
                "type": "box",
                "size": "0.045 0.040 0.090",
                "rgba": "0.94 0.48 0.08 1",
                "contype": "0",
                "conaffinity": "0",
                "group": "0",
            },
        )
        worldbody.insert(4, pickup_object)

    # assembly.xml also contains a legacy red `eef` site. It is only a visual
    # marker: the policy reads the link6 body pose, so hiding it cannot change
    # the observation or controller state.
    if hide_current_ee_frame:
        legacy_eef_site = root.find(".//site[@name='eef']")
        if legacy_eef_site is not None:
            legacy_eef_site.set("rgba", "1 0.2 0.1 0")

    # The measured frame is shorter and translucent. It can be omitted from
    # the generated MJCF explicitly without changing link6 or policy inputs.
    if not hide_current_ee_frame:
        eef_frame = worldbody.find(".//body[@name='link6']")
        if eef_frame is None:
            raise ValueError("J6 child body link6 is missing from MJCF")
        ET.SubElement(
            eef_frame,
            "site",
            {
                "name": "current_eef",
                "type": "sphere",
                "size": "0.005",
                "rgba": "0.88 0.92 0.98 0.65",
                "group": "0",
            },
        )
        for name, endpoint, color in (
            ("current_eef_x", "0.08 0 0", "0.95 0.20 0.16 0.58"),
            ("current_eef_y", "0 0.08 0", "0.10 0.72 0.34 0.58"),
            ("current_eef_z", "0 0 0.08", "0.12 0.38 0.92 0.58"),
        ):
            ET.SubElement(
                eef_frame,
                "site",
                {
                    "name": name,
                    "type": "capsule",
                    "fromto": f"0 0 0 {endpoint}",
                    "size": "0.0025",
                    "rgba": color,
                    "group": "0",
                },
            )
            ET.SubElement(
                eef_frame,
                "site",
                {
                    "name": f"{name}_tip",
                    "type": "sphere",
                    "pos": endpoint,
                    "size": "0.0045",
                    "rgba": color,
                    "group": "0",
                },
            )

    # assembly.urdf models the UMI gripper as fixed geometry attached to
    # link6. Do not inject the obsolete six-DoF DAS gripper at runtime.

    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml)


class ThreeOnnxPolicy:
    def __init__(self, model_dir: Path, sample_latent: bool, rng: np.random.Generator):
        model_dir = model_dir.expanduser().resolve()
        missing = [
            name for name in ("actor.onnx", "contactNet.onnx", "gru.onnx") if not (model_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {missing} in {model_dir}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.actor = ort.InferenceSession(str(model_dir / "actor.onnx"), options, providers=providers)
        self.contact_net = ort.InferenceSession(
            str(model_dir / "contactNet.onnx"), options, providers=providers
        )
        self.gru = ort.InferenceSession(str(model_dir / "gru.onnx"), options, providers=providers)
        self.sample_latent = sample_latent
        self.rng = rng

        actor_inputs = self.actor.get_inputs()
        contact_input = self.contact_net.get_inputs()[0]
        gru_inputs = self.gru.get_inputs()
        if actor_inputs[0].shape[-1] != OBS_DIM or actor_inputs[1].shape[-1] != 67:
            raise ValueError(f"Unexpected actor inputs: {[item.shape for item in actor_inputs]}")
        if contact_input.shape[-1] != CONTACT_OBS_DIM:
            raise ValueError(f"Unexpected contactNet input: {contact_input.shape}")
        if gru_inputs[0].shape[-1] != 131 or gru_inputs[1].shape[-1] != 131:
            raise ValueError(f"Unexpected GRU inputs: {[item.shape for item in gru_inputs]}")

        # The contactNet export advertises a dynamic sequence axis, but its
        # attention reshape is trained/exported for exactly ten frames. Probe
        # the complete three-model contract now so a mismatched export fails
        # before the simulation starts.
        contact_probe = np.zeros((1, HISTORY_LENGTH, CONTACT_OBS_DIM), dtype=np.float32)
        contact_output = self.contact_net.run(
            None,
            {contact_input.name: contact_probe},
        )[0]
        if np.asarray(contact_output).shape != (1, 131):
            raise ValueError(
                "contactNet.onnx does not satisfy the required "
                f"[1,{HISTORY_LENGTH},{CONTACT_OBS_DIM}] -> [1,131] contract: "
                f"got {np.asarray(contact_output).shape}"
            )
        hidden_probe = np.zeros((1, 1, 131), dtype=np.float32)
        gru_output, new_hidden = self.gru.run(
            None,
            {
                gru_inputs[0].name: np.asarray(contact_output, dtype=np.float32),
                gru_inputs[1].name: hidden_probe,
            },
        )
        if np.asarray(gru_output).shape != (1, 131) or np.asarray(new_hidden).shape != (1, 1, 131):
            raise ValueError(
                "gru.onnx does not satisfy the required [1,131] + [1,1,131] contract"
            )
        actor_probe = self.actor.run(
            None,
            {
                actor_inputs[0].name: np.zeros((1, OBS_DIM), dtype=np.float32),
                actor_inputs[1].name: np.zeros((1, 67), dtype=np.float32),
            },
        )[0]
        if np.asarray(actor_probe).shape != (1, ACTION_DIM):
            raise ValueError(
                f"actor.onnx does not satisfy the required [1,67]+[1,67]->[1,14] contract: "
                f"got {np.asarray(actor_probe).shape}"
            )

        self.hidden = np.zeros((1, 1, 131), dtype=np.float32)

    def reset(self) -> None:
        self.hidden.fill(0.0)

    def __call__(self, observation: np.ndarray, history: np.ndarray) -> np.ndarray:
        contact_input = history[np.newaxis, :, :].astype(np.float32, copy=False)
        contact_name = self.contact_net.get_inputs()[0].name
        contact_output = self.contact_net.run(None, {contact_name: contact_input})[0]
        contact_output = np.asarray(contact_output[-1:], dtype=np.float32)

        gru_inputs = self.gru.get_inputs()
        gru_output, new_hidden = self.gru.run(
            None,
            {
                gru_inputs[0].name: contact_output,
                gru_inputs[1].name: self.hidden,
            },
        )
        self.hidden = np.asarray(new_hidden, dtype=np.float32)
        gru_output = np.asarray(gru_output, dtype=np.float32)

        # GRU output = [base_lin_vel(3), mu(64), logvar(64)].
        mean = gru_output[:, 3:67]
        if self.sample_latent:
            log_variance = gru_output[:, 67:131]
            std = np.sqrt(np.exp(log_variance) + 1.0e-4)
            predicted = mean + std * self.rng.standard_normal(mean.shape).astype(np.float32)
        else:
            predicted = mean
        actor_latent = np.concatenate((gru_output[:, :3], predicted), axis=1).astype(np.float32)

        actor_inputs = self.actor.get_inputs()
        action = self.actor.run(
            None,
            {
                actor_inputs[0].name: observation[np.newaxis, :].astype(np.float32),
                actor_inputs[1].name: actor_latent,
            },
        )[0]
        action = np.asarray(action[0], dtype=np.float64)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid actor output: shape={action.shape}, values={action}")
        return action


class TerminalKeyboardController:
    """Read target commands directly from the launching terminal."""

    POSITION_KEY_DELTAS = {
        "W": np.array([1.0, 0.0, 0.0]),
        "S": np.array([-1.0, 0.0, 0.0]),
        "A": np.array([0.0, 1.0, 0.0]),
        "D": np.array([0.0, -1.0, 0.0]),
        "R": np.array([0.0, 0.0, 1.0]),
        "F": np.array([0.0, 0.0, -1.0]),
    }
    RPY_KEY_DELTAS = {
        "T": np.array([1.0, 0.0, 0.0]),
        "G": np.array([-1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "H": np.array([0.0, -1.0, 0.0]),
        "U": np.array([0.0, 0.0, 1.0]),
        "J": np.array([0.0, 0.0, -1.0]),
    }

    def __init__(self, position_step: float, rotation_step: float):
        if not math.isfinite(position_step) or position_step <= 0:
            raise ValueError("--keyboard-step must be greater than zero")
        if not math.isfinite(rotation_step) or rotation_step <= 0:
            raise ValueError("--keyboard-rotation-step must be greater than zero")
        self.position_step = float(position_step)
        self.rotation_step = float(rotation_step)
        self._pending_position_delta = np.zeros(3, dtype=np.float64)
        self._pending_rpy_delta = np.zeros(3, dtype=np.float64)
        self._print_requested = False
        self._gripper_command: float | None = None
        self._quit_requested = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fd: int | None = None
        self._saved_terminal_settings = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Enter cbreak mode so each terminal key is available immediately."""
        if not sys.stdin.isatty():
            print("[keyboard] stdin 不是终端；终端监听已禁用，MuJoCo 窗口键盘控制仍可用。")
            return False
        self._fd = sys.stdin.fileno()
        self._saved_terminal_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(
            target=self._read_loop,
            name="sim2sim-terminal-keyboard",
            daemon=True,
        )
        self._thread.start()
        return True

    def _read_loop(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if not readable:
                    continue
                raw_key = os.read(self._fd, 1)
            except OSError:
                break
            if not raw_key:
                break
            self._queue_key(raw_key.decode(errors="ignore"))

    def _queue_key(self, key: str) -> None:
        """Queue one target-control key from either terminal or viewer."""
        key = key.upper()
        with self._lock:
            position_direction = self.POSITION_KEY_DELTAS.get(key)
            rotation_direction = self.RPY_KEY_DELTAS.get(key)
            if position_direction is not None:
                self._pending_position_delta += position_direction * self.position_step
            elif rotation_direction is not None:
                self._pending_rpy_delta += rotation_direction * self.rotation_step
            elif key == "P":
                self._print_requested = True
            elif key == "O":
                self._gripper_command = 0.0
            elif key == "C":
                self._gripper_command = 1.0
            elif key == "Q":
                self._quit_requested = True

    def viewer_key_callback(self, keycode: int) -> None:
        """Receive GLFW key codes while the MuJoCo viewer has focus."""
        if 0 <= keycode <= 0x10FFFF:
            self._queue_key(chr(keycode))

    def consume(self) -> tuple[np.ndarray, np.ndarray, bool, float | None, bool]:
        with self._lock:
            position_delta = self._pending_position_delta.copy()
            rpy_delta = self._pending_rpy_delta.copy()
            print_requested = self._print_requested
            gripper_command = self._gripper_command
            quit_requested = self._quit_requested
            self._pending_position_delta.fill(0.0)
            self._pending_rpy_delta.fill(0.0)
            self._print_requested = False
            self._gripper_command = None
        return position_delta, rpy_delta, print_requested, gripper_command, quit_requested

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._fd is not None and self._saved_terminal_settings is not None:
            termios.tcsetattr(
                self._fd,
                termios.TCSADRAIN,
                self._saved_terminal_settings,
            )
            self._saved_terminal_settings = None


class Sim2Sim:
    def __init__(
        self,
        model: mujoco.MjModel,
        policy: ThreeOnnxPolicy,
        command: np.ndarray,
        command_frame: str,
        base_height: float,
        arm_max_step: float,
        max_leg_step: float,
        lock_j6: bool,
        lock_j6_angle: float,
        policy_decimation: int,
        se3_decrease_vel: float,
        action_scale: np.ndarray,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.policy = policy
        self.command_frame = command_frame
        self.target_position = command[:3].copy()
        self.target_rotation = rotation_from_rpy(*command[3:])
        if not math.isfinite(arm_max_step) or arm_max_step < 0.0:
            raise ValueError("--arm-max-step must be finite and non-negative")
        if not math.isfinite(max_leg_step) or max_leg_step < 0.0:
            raise ValueError("--max-leg-step must be finite and non-negative")
        self.arm_max_step = float(arm_max_step)
        self.max_leg_step = float(max_leg_step)
        if not math.isfinite(lock_j6_angle):
            raise ValueError("--lock-j6-angle must be finite")
        self.lock_j6 = bool(lock_j6)
        self.lock_j6_angle = float(lock_j6_angle)
        j6_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "J6")
        if j6_joint_id < 0:
            raise ValueError("J6 joint is missing from the MJCF")
        if self.lock_j6 and model.jnt_limited[j6_joint_id]:
            j6_min, j6_max = model.jnt_range[j6_joint_id]
            if not j6_min <= self.lock_j6_angle <= j6_max:
                raise ValueError(
                    f"--lock-j6-angle={self.lock_j6_angle:g} is outside "
                    f"J6 range [{j6_min:g}, {j6_max:g}]"
                )
        self.policy_decimation = int(policy_decimation)
        if not math.isfinite(se3_decrease_vel) or se3_decrease_vel < 0.0:
            raise ValueError("--se3-decrease-vel must be finite and non-negative")
        self.se3_decrease_vel = float(se3_decrease_vel)
        self.action_scale = np.asarray(action_scale, dtype=np.float64).copy()
        if self.action_scale.shape != (ACTION_DIM,):
            raise ValueError(f"action_scale must have shape ({ACTION_DIM},)")
        if not np.isfinite(self.action_scale).all() or np.any(self.action_scale <= 0.0):
            raise ValueError("all action scale values must be finite and positive")

        self.joint_qpos_adr = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=int)
        self.joint_dof_adr = np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES], dtype=int)
        self.motor_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor") for name in JOINT_NAMES],
            dtype=int,
        )
        if np.any(self.motor_ids < 0):
            raise ValueError("One or more joint motors are missing from the MJCF")
        self.gripper_command = 0.0

        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        self.target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "command_target")
        target_frame_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "command_target_frame"
        )
        self.target_mocap_id = (
            int(model.body_mocapid[target_frame_body_id]) if target_frame_body_id >= 0 else -1
        )
        self.imu_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if min(
            self.base_body_id,
            self.ee_body_id,
            self.target_site_id,
            self.target_mocap_id,
            self.imu_sensor_id,
        ) < 0:
            raise ValueError("Required base/EE/IMU/target elements are missing")

        mujoco.mj_resetData(model, self.data)
        root_adr = model.joint("root").qposadr[0]
        self.data.qpos[root_adr : root_adr + 7] = [0.0, 0.0, base_height, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.joint_qpos_adr] = ISAAC_RESET_JOINT_POS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(model, self.data)

        self.raw_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.effective_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        # Before the first action is processed, IsaacLab's implicit actuators
        # still have zero position targets.  Its initial applied_torque history
        # is therefore KP * (0 - q_reset), notably J2=-9 and J3=-1.17 Nm.
        # Seed all ten contactNet frames with the same reset-time semantics.
        self.last_torque = np.clip(
            -KP * ISAAC_RESET_JOINT_POS,
            -TORQUE_LIMIT,
            TORQUE_LIMIT,
        )
        self.raw_desired_position = DEFAULT_JOINT_POS.copy()
        self.policy_desired_position = DEFAULT_JOINT_POS.copy()
        self.desired_position = DEFAULT_JOINT_POS.copy()
        self.policy_command_step = np.zeros(ACTION_DIM, dtype=np.float64)
        self._previous_policy_command = DEFAULT_JOINT_POS.copy()
        self.history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        self.se3_distance_reference = self._initial_se3_distance()
        first_contact_obs = self.contact_observation()
        for _ in range(HISTORY_LENGTH):
            self.history.append(first_contact_obs.copy())
        self.policy.reset()
        self._update_target_marker()

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position = self.data.xpos[self.base_body_id].copy()
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3).copy()
        return position, rotation

    def ee_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        base_position, base_rotation = self.base_pose()
        ee_position_world = self.data.xpos[self.ee_body_id]
        ee_rotation_world = self.data.xmat[self.ee_body_id].reshape(3, 3)
        position = base_rotation.T @ (ee_position_world - base_position)
        rotation = base_rotation.T @ ee_rotation_world
        return position, rotation

    def target_pose_base(self) -> tuple[np.ndarray, np.ndarray]:
        if self.command_frame == "base":
            return self.target_position, self.target_rotation
        base_position, base_rotation = self.base_pose()
        position = base_rotation.T @ (self.target_position - base_position)
        rotation = base_rotation.T @ self.target_rotation
        return position, rotation

    def target_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        if self.command_frame == "world":
            return self.target_position, self.target_rotation
        base_position, base_rotation = self.base_pose()
        return (
            base_position + base_rotation @ self.target_position,
            base_rotation @ self.target_rotation,
        )

    @staticmethod
    def pose_6d(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        return np.concatenate((position, rotation[:, 0], rotation[:, 1]))

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.data.qpos[self.joint_qpos_adr].copy(),
            self.data.qvel[self.joint_dof_adr].copy(),
        )

    def base_angular_velocity(self) -> np.ndarray:
        sensor = self.model.sensor(self.imu_sensor_id)
        start = sensor.adr[0]
        return self.data.sensordata[start : start + 3].copy()

    def projected_gravity(self) -> np.ndarray:
        _, base_rotation = self.base_pose()
        return base_rotation.T @ np.array([0.0, 0.0, -1.0])

    def contact_observation(self) -> np.ndarray:
        position, velocity = self.joint_state()
        ee_position, ee_rotation = self.ee_pose_base()
        observation = np.concatenate(
            (
                self.base_angular_velocity(),
                self.projected_gravity(),
                position - DEFAULT_JOINT_POS,
                velocity,
                self.last_torque,
                self.pose_6d(ee_position, ee_rotation),
            )
        )
        if observation.shape != (CONTACT_OBS_DIM,):
            raise RuntimeError(f"Contact observation has shape {observation.shape}")
        return observation

    def policy_observation(self) -> np.ndarray:
        position, velocity = self.joint_state()
        ee_position, ee_rotation = self.ee_pose_base()
        target_position, target_rotation = self.target_pose_base()
        observation = np.concatenate(
            (
                self.base_angular_velocity(),
                self.projected_gravity(),
                self.pose_6d(target_position, target_rotation),
                position - DEFAULT_JOINT_POS,
                velocity,
                self.last_action,
                self.pose_6d(ee_position, ee_rotation),
                np.array([self.se3_distance_reference]),
            )
        )
        if observation.shape != (OBS_DIM,):
            raise RuntimeError(f"Policy observation has shape {observation.shape}")
        return observation

    def _initial_se3_distance(self) -> float:
        ee_position, ee_rotation = self.ee_pose_base()
        target_position, target_rotation = self.target_pose_base()
        return float(
            2.0 * np.linalg.norm(target_position - ee_position)
            + rotation_angle(target_rotation @ ee_rotation.T)
        )

    def _update_target_marker(self) -> None:
        target_position, target_rotation = self.target_pose_world()
        target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(target_quaternion, target_rotation.reshape(-1))
        self.data.mocap_pos[self.target_mocap_id] = target_position
        self.data.mocap_quat[self.target_mocap_id] = target_quaternion

    def translate_target(self, delta: np.ndarray) -> None:
        """Translate the target in the selected command frame."""
        self.target_position += np.asarray(delta, dtype=np.float64)
        self.se3_distance_reference = self._initial_se3_distance()
        self._update_target_marker()

    def rotate_target(self, delta_rpy: np.ndarray) -> None:
        """Rotate the target incrementally about its local roll/pitch/yaw axes."""
        delta_rpy = np.asarray(delta_rpy, dtype=np.float64)
        if delta_rpy.shape != (3,) or not np.isfinite(delta_rpy).all():
            raise ValueError("Target RPY delta must contain three finite values")
        self.target_rotation = self.target_rotation @ rotation_from_rpy(*delta_rpy)
        self.se3_distance_reference = self._initial_se3_distance()
        self._update_target_marker()

    def set_target(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        *,
        reset_reference: bool = False,
    ) -> None:
        self.target_position = np.asarray(position, dtype=np.float64).copy()
        self.target_rotation = np.asarray(rotation, dtype=np.float64).copy()
        if reset_reference:
            self.se3_distance_reference = self._initial_se3_distance()
        self._update_target_marker()

    def set_gripper_command(self, command: float) -> None:
        """Keep keyboard compatibility; the URDF gripper itself is fixed."""
        self.gripper_command = float(np.clip(command, 0.0, 1.0))

    def gripper_position(self) -> float:
        """The current URDF has no gripper DoF."""
        return 0.0

    def infer(self) -> None:
        contact_obs = self.contact_observation()
        self.history.append(contact_obs)
        history = np.stack(self.history, axis=0)
        self.raw_action = self.policy(self.policy_observation(), history)
        self.se3_distance_reference = max(
            0.0,
            self.se3_distance_reference - self.se3_decrease_vel * POLICY_DT,
        )

    def apply_pd(self, *, policy_updated: bool = False) -> None:
        position, velocity = self.joint_state()
        if policy_updated:
            # Match IsaacLab's JointPositionAction + ImplicitActuator chain:
            # keep the raw position target for the full policy interval and
            # saturate only the resulting PD torque below.  Pre-clipping the
            # position target changes both the closed-loop dynamics and the
            # last_action observation seen by the policy.
            scaled_action = self.raw_action * self.action_scale
            self.raw_desired_position = DEFAULT_JOINT_POS + scaled_action
            self.policy_desired_position = self.raw_desired_position.copy()

            command = self.policy_desired_position.copy()
            if self.arm_max_step > 0.0:
                command[ARM_IDS] = np.clip(
                    command[ARM_IDS],
                    self._previous_policy_command[ARM_IDS] - self.arm_max_step,
                    self._previous_policy_command[ARM_IDS] + self.arm_max_step,
                )
            if self.max_leg_step > 0.0:
                command[LEG_IDS] = np.clip(
                    command[LEG_IDS],
                    self._previous_policy_command[LEG_IDS] - self.max_leg_step,
                    self._previous_policy_command[LEG_IDS] + self.max_leg_step,
                )
            if self.lock_j6:
                command[J6_ID] = self.lock_j6_angle
            self.desired_position = command
            self.effective_action = command - DEFAULT_JOINT_POS
            self.policy_command_step = command - self._previous_policy_command
            self._previous_policy_command = command.copy()
            # mdp.last_action in IsaacLab exposes the raw action-manager input,
            # not a torque-aware or rate-limited target.
            self.last_action = self.raw_action.copy()
            if self.lock_j6:
                # Match the deployment controller: the policy observes the
                # effective fixed-position action for a locked J6.
                self.last_action[J6_ID] = (
                    self.lock_j6_angle - DEFAULT_JOINT_POS[J6_ID]
                ) / self.action_scale[J6_ID]

        torque = KP * (self.desired_position - position) - KD * velocity
        torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.motor_ids] = torque
        self.last_torque = torque

    def step(self, physics_step: int) -> None:
        policy_updated = physics_step % self.policy_decimation == 0
        if policy_updated:
            self.infer()
        self.apply_pd(policy_updated=policy_updated)
        mujoco.mj_step(self.model, self.data)
        # PhysX applies the ImplicitActuator velocity limits configured by the
        # training asset. MuJoCo hinge joints have no equivalent max-velocity
        # field, so enforce the same state constraint after integration.
        self.data.qvel[self.joint_dof_adr] = np.clip(
            self.data.qvel[self.joint_dof_adr],
            -VELOCITY_LIMIT,
            VELOCITY_LIMIT,
        )
        self._update_target_marker()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Simulation state became non-finite")

    def error(self) -> tuple[float, float]:
        ee_position, ee_rotation = self.ee_pose_base()
        target_position, target_rotation = self.target_pose_base()
        return (
            float(np.linalg.norm(target_position - ee_position)),
            rotation_angle(target_rotation @ ee_rotation.T),
        )


def configure_viewer(
    viewer_handle,
    base_body_id: int,
    track_robot: bool,
    camera_azimuth: float,
    camera_elevation: float,
    camera_distance: float,
) -> None:
    if track_robot:
        viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer_handle.cam.trackbodyid = base_body_id
        viewer_handle.cam.fixedcamid = -1
    else:
        viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer_handle.cam.trackbodyid = -1
    viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.45]
    viewer_handle.cam.distance = camera_distance
    viewer_handle.cam.azimuth = camera_azimuth
    viewer_handle.cam.elevation = camera_elevation
    viewer_handle.opt.geomgroup[2] = 1
    viewer_handle.opt.geomgroup[3] = 0


def run(args: argparse.Namespace) -> None:
    if not math.isfinite(args.physics_dt) or args.physics_dt <= 0.0:
        raise ValueError("--physics-dt must be finite and positive")
    if args.freeze_after is not None and (
        not math.isfinite(args.freeze_after) or args.freeze_after < 0.0
    ):
        raise ValueError("--freeze-after must be finite and non-negative")
    decimation_float = POLICY_DT / args.physics_dt
    policy_decimation = round(decimation_float)
    if policy_decimation < 1 or not math.isclose(
        decimation_float, policy_decimation, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("--physics-dt must divide the 0.02 s policy period exactly")

    trajectory = None
    if args.trajectory is not None:
        trajectory = PickleTrajectory(
            args.trajectory,
            args.trajectory_index,
            args.trajectory_start_delay,
            args.trajectory_loop,
            planar_center=not args.no_planar_center,
            playback_speed=args.trajectory_speed,
        )
        if args.command is not None:
            command = np.asarray(args.command, dtype=np.float64)
            command_frame = args.command_frame
        elif trajectory.position_mode in ("absolute_world", "absolute_base"):
            command = np.concatenate(
                (trajectory.positions[0], rpy_from_rotation(trajectory.rotations[0]))
            )
            command_frame = (
                "base" if trajectory.position_mode == "absolute_base" else "world"
            )
        else:
            # Fallback hold target before a relative trajectory anchors itself.
            command = np.array([0.15, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            command_frame = "world"
    else:
        command = read_command(args.command)
        command_frame = args.command_frame
    if (args.pickup_object or args.payload_gravity) and trajectory is None:
        raise ValueError("--pickup-object and --payload-gravity require --trajectory")
    if not math.isfinite(args.record_smoothing) or args.record_smoothing < 0.0:
        raise ValueError("--record-smoothing must be finite and non-negative")

    record_output: Path | None = None
    raw_trajectory_output: Path | None = None
    smooth_trajectory_output: Path | None = None
    if args.record_output is not None:
        record_output = args.record_output.expanduser().resolve()
        if record_output.suffix.lower() != ".npz":
            raise ValueError("--record-output must end in .npz")
        raw_trajectory_output = record_output.with_name(
            f"{record_output.stem}_raw.pkl"
        )
        smooth_trajectory_output = record_output.with_name(
            f"{record_output.stem}_smooth.pkl"
        )
        existing = [
            path
            for path in (record_output, raw_trajectory_output, smooth_trajectory_output)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite an existing keyboard recording: "
                + ", ".join(str(path) for path in existing)
            )

    model_dir = args.model_dir or newest_exported_model_dir()
    rng = np.random.default_rng(args.seed)
    if args.action_scale is None:
        action_scale = ACTION_SCALE.copy()
        action_scale_description = "legs/ankles 0.6, J1-J3 0.3, J4-J6 0.2"
    else:
        if not math.isfinite(args.action_scale) or args.action_scale <= 0.0:
            raise ValueError("--action-scale must be finite and positive")
        action_scale = np.full(ACTION_DIM, args.action_scale, dtype=np.float64)
        action_scale_description = f"all joints {args.action_scale:g}"

    print(f"[sim2sim] MJCF: {args.mjcf.expanduser().resolve()}")
    print(f"[sim2sim] ONNX: {model_dir.expanduser().resolve()}")
    print(
        "[sim2sim] 训练契约：EE frame=link6；"
        f"action scale={action_scale_description}"
    )
    print(
        "[sim2sim] 最终点 "
        f"({command_frame}): xyz={command[:3].tolist()}, "
        f"rpy={command[3:].tolist()} rad"
    )
    print(
        "[sim2sim] 50Hz目标限幅："
        f"arm_max_step={args.arm_max_step:g} rad，"
        f"max_leg_step={args.max_leg_step:g} rad（0=关闭）"
    )
    print(
        "[sim2sim] J6："
        + (
            f"强制锁定在 {args.lock_j6_angle:g} rad"
            if args.lock_j6
            else "由策略控制"
        )
    )
    clock_status = (
        "严格匹配训练"
        if math.isclose(args.physics_dt, TRAINING_PHYSICS_DT)
        else "MuJoCo稳定接触子步（策略周期仍匹配训练）"
    )
    print(
        f"[sim2sim] 时钟：physics_dt={args.physics_dt:g}s × {policy_decimation} "
        f"= policy_dt={POLICY_DT:g}s；{clock_status}"
    )
    if math.isclose(args.physics_dt, TRAINING_PHYSICS_DT):
        print(
            "[sim2sim] WARNING: 5 ms 与训练时钟相同，但当前 MuJoCo 足底 mesh "
            "长时间接触可能不稳定；PhysX 与 MuJoCo 求解器仍不会逐值相同。"
        )
    print(f"[sim2sim] 场景：{'平地 + 门（部署测试）' if args.door else '平地（匹配训练）'}")
    print(
        f"[sim2sim] SE3参考衰减：{args.se3_decrease_vel:g} m/rad-equivalent per second"
    )
    print(
        f"[sim2sim] latent：{'逐步采样（匹配训练）' if args.sample_latent else '使用均值（确定性诊断）'}"
    )
    if args.freeze_after is not None:
        print(
            f"[sim2sim] 冻结：运行到 {args.freeze_after:g}s 后停止物理并保持窗口"
        )
    if command_frame == "base":
        print(
            "[sim2sim] WARNING: --command-frame base 会让目标跟随移动基座；"
            "训练契约是采样后固定在 world，严格对比请使用默认 world。"
        )
    if trajectory is not None:
        print(
            f"[trajectory] file={trajectory.path}, episode={trajectory.episode_index}, "
            f"frames={len(trajectory.positions)}, sample_dt={trajectory.sample_dt:g}s, "
            f"speed={trajectory.playback_speed:g}x, "
            f"duration={trajectory.duration:.3f}s, start_delay={trajectory.start_delay:g}s, "
            f"position_mode={trajectory.position_mode}"
        )

    model = load_sim_model(
        args.mjcf,
        physics_dt=args.physics_dt,
        add_door=args.door,
        hide_current_ee_frame=args.hide_current_ee_frame,
        add_pickup_object=args.pickup_object,
    )
    policy = ThreeOnnxPolicy(model_dir, args.sample_latent, rng)
    simulation = Sim2Sim(
        model,
        policy,
        command,
        command_frame,
        args.base_height,
        args.arm_max_step,
        args.max_leg_step,
        args.lock_j6,
        args.lock_j6_angle,
        policy_decimation,
        args.se3_decrease_vel,
        action_scale,
    )
    pickup_mocap_id = -1
    if args.pickup_object:
        assert trajectory is not None
        pickup_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "sim2sim_pickup_object"
        )
        if pickup_body_id < 0:
            raise ValueError("pickup object was requested but is missing from the model")
        pickup_mocap_id = int(model.body_mocapid[pickup_body_id])
        simulation.data.mocap_pos[pickup_mocap_id] = trajectory.object_position_world
        simulation.data.mocap_quat[pickup_mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, simulation.data)
    if args.eef_x_offset is not None:
        if not math.isfinite(args.eef_x_offset):
            raise ValueError("--eef-x-offset must be finite")
        initial_eef_world = simulation.data.xpos[simulation.ee_body_id].copy()
        initial_eef_rotation_world = simulation.data.xmat[simulation.ee_body_id].reshape(3, 3).copy()
        target_eef_world = initial_eef_world + np.array(
            [args.eef_x_offset, 0.0, 0.0], dtype=np.float64
        )
        simulation.command_frame = "world"
        simulation.set_target(
            target_eef_world,
            initial_eef_rotation_world,
            reset_reference=True,
        )
        actual_offset = simulation.target_position - initial_eef_world
        print(
            "[sim2sim] EEF相对目标确认："
            f"initial_world={initial_eef_world.round(6).tolist()}, "
            f"target_world={simulation.target_position.round(6).tolist()}, "
            f"delta_world={actual_offset.round(6).tolist()} m"
        )
    keyboard = TerminalKeyboardController(args.keyboard_step, args.keyboard_rotation_step)
    record: dict[str, list] = {
        "time": [],
        "target_position_base": [],
        "target_rotation_base": [],
        "measured_position_base": [],
        "measured_rotation_base": [],
        "target_position_world": [],
        "target_rotation_world": [],
        "measured_position_world": [],
        "measured_rotation_world": [],
        "base_position_world": [],
        "joint_position": [],
        "joint_velocity": [],
        "joint_torque": [],
        "raw_action": [],
        "effective_action": [],
        "gripper_command": [],
    }
    if record_output is not None:
        print(f"[record] keyboard raw data: {record_output}")
        print(f"[record] raw trajectory: {raw_trajectory_output}")
        print(
            f"[record] smoothed trajectory: {smooth_trajectory_output} "
            f"(Gaussian sigma={args.record_smoothing:g}s)"
        )

    if args.duration is None:
        if trajectory is None:
            run_duration = 10.0
        elif trajectory.loop:
            run_duration = 0.0
        else:
            # Include one more policy tick so the final recorded frame is
            # applied before an automatic non-looping run exits.
            run_duration = trajectory.start_delay + trajectory.duration + POLICY_DT
    else:
        run_duration = args.duration
    if run_duration > 10.0 and trajectory is None:
        print(
            "[sim2sim] WARNING: 运行时间超过训练 episode_length_s=10；"
            "10 秒后的递归状态不再严格属于训练时序。"
        )
    max_steps = math.inf if run_duration <= 0 else math.ceil(run_duration / args.physics_dt)
    log_steps = max(1, round(args.log_interval / args.physics_dt))
    render_steps = max(1, round(1.0 / (max(args.render_fps, 1.0) * args.physics_dt)))

    def print_keyboard_target() -> None:
        target_rpy = rpy_from_rotation(simulation.target_rotation)
        print(
            f"[keyboard] target({simulation.command_frame}) "
            f"xyz={simulation.target_position.round(4).tolist()}, "
            f"rpy={target_rpy.round(4).tolist()} rad"
        )

    def loop(viewer_handle=None) -> None:
        physics_step = 0
        freeze_message_printed = False
        # Establish the wall-clock epoch only after the viewer has finished
        # opening. Otherwise its startup cost makes the simulation briefly
        # race ahead in an attempt to "catch up".
        wall_epoch = time.perf_counter() - simulation.data.time
        last_log_wall = time.perf_counter()
        last_log_sim = simulation.data.time
        viewer_sync_count = 0
        while physics_step < max_steps and (viewer_handle is None or viewer_handle.is_running()):
            policy_updated = physics_step % policy_decimation == 0
            (
                target_delta,
                target_rpy_delta,
                print_target,
                gripper_command,
                quit_requested,
            ) = keyboard.consume()
            if quit_requested:
                print("\n[keyboard] 终端请求退出。")
                break
            if (
                args.freeze_after is not None
                and simulation.data.time >= args.freeze_after
            ):
                if not freeze_message_printed:
                    print(
                        f"[sim2sim] 已在 t={simulation.data.time:.3f}s 冻结；"
                        "关闭窗口或按 Q 退出。"
                    )
                    freeze_message_printed = True
                if viewer_handle is None:
                    break
                viewer_handle.sync()
                time.sleep(1.0 / max(args.render_fps, 1.0))
                continue
            if gripper_command is not None:
                simulation.set_gripper_command(gripper_command)
                print("[keyboard] 当前 URDF 将夹爪建模为固定几何，O/C 不改变模型。")
            target_changed = False
            if np.any(target_delta):
                if trajectory is not None:
                    trajectory.translate_offset(target_delta)
                    print(
                        "[keyboard] trajectory world offset="
                        f"{trajectory.world_offset.round(4).tolist()}"
                    )
                else:
                    simulation.translate_target(target_delta)
                    target_changed = True
            if np.any(target_rpy_delta):
                if trajectory is not None:
                    print("[keyboard] 轨迹模式暂不支持姿态偏移；RPY 按键已忽略。")
                else:
                    simulation.rotate_target(target_rpy_delta)
                    target_changed = True
            if target_changed or print_target:
                print_keyboard_target()
            if trajectory is not None and policy_updated:
                trajectory.update(simulation)
            simulation.data.qfrc_applied[:] = 0.0
            payload_mass = 0.0
            if trajectory is not None:
                payload_mass = float(trajectory.payload_mass[trajectory.current_frame])
                ee_position_world = simulation.data.xpos[simulation.ee_body_id].copy()
                ee_rotation_world = simulation.data.xmat[simulation.ee_body_id].reshape(3, 3).copy()
                pickup_point_world = (
                    ee_position_world + ee_rotation_world @ trajectory.pickup_tip_offset
                )
                if args.payload_gravity and payload_mass > 0.0:
                    mujoco.mj_applyFT(
                        model,
                        simulation.data,
                        np.array([0.0, 0.0, -payload_mass * 9.81], dtype=np.float64),
                        np.zeros(3, dtype=np.float64),
                        pickup_point_world,
                        simulation.ee_body_id,
                        simulation.data.qfrc_applied,
                    )
                if pickup_mocap_id >= 0:
                    object_grasped = (
                        trajectory.current_frame * trajectory.sample_dt
                        >= trajectory.grasp_time
                    )
                    if object_grasped:
                        simulation.data.mocap_pos[pickup_mocap_id] = pickup_point_world
                        pickup_quaternion = np.empty(4, dtype=np.float64)
                        mujoco.mju_mat2Quat(
                            pickup_quaternion, ee_rotation_world.reshape(-1)
                        )
                        simulation.data.mocap_quat[pickup_mocap_id] = pickup_quaternion
                    else:
                        simulation.data.mocap_pos[pickup_mocap_id] = (
                            trajectory.object_position_world
                        )
                        simulation.data.mocap_quat[pickup_mocap_id] = [1.0, 0.0, 0.0, 0.0]
            simulation.step(physics_step)
            if policy_updated and record_output is not None:
                target_position_base, target_rotation_base = simulation.target_pose_base()
                measured_position_base, measured_rotation_base = simulation.ee_pose_base()
                target_position_world, target_rotation_world = simulation.target_pose_world()
                measured_position_world = simulation.data.xpos[
                    simulation.ee_body_id
                ].copy()
                measured_rotation_world = simulation.data.xmat[
                    simulation.ee_body_id
                ].reshape(3, 3).copy()
                joint_position, joint_velocity = simulation.joint_state()
                record["time"].append(float(simulation.data.time))
                record["target_position_base"].append(target_position_base.copy())
                record["target_rotation_base"].append(target_rotation_base.copy())
                record["measured_position_base"].append(measured_position_base.copy())
                record["measured_rotation_base"].append(measured_rotation_base.copy())
                record["target_position_world"].append(target_position_world.copy())
                record["target_rotation_world"].append(target_rotation_world.copy())
                record["measured_position_world"].append(measured_position_world)
                record["measured_rotation_world"].append(measured_rotation_world)
                record["base_position_world"].append(
                    simulation.data.xpos[simulation.base_body_id].copy()
                )
                record["joint_position"].append(joint_position)
                record["joint_velocity"].append(joint_velocity)
                record["joint_torque"].append(simulation.last_torque.copy())
                record["raw_action"].append(simulation.raw_action.copy())
                record["effective_action"].append(simulation.effective_action.copy())
                record["gripper_command"].append(float(simulation.gripper_command))
            physics_step += 1
            # Physics runs at 1 kHz, but synchronizing the GUI at 1 kHz makes
            # wall-clock time lag badly and looks like slow motion.
            if viewer_handle is not None and physics_step % render_steps == 0:
                viewer_handle.sync()
                viewer_sync_count += 1
            if not args.no_realtime:
                deadline = wall_epoch + simulation.data.time
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            if physics_step % log_steps == 0:
                now = time.perf_counter()
                wall_delta = max(now - last_log_wall, 1.0e-9)
                sim_delta = simulation.data.time - last_log_sim
                rtf = sim_delta / wall_delta
                viewer_fps = viewer_sync_count / wall_delta if viewer_handle is not None else 0.0
                pos_error, rot_error = simulation.error()
                base_z = simulation.data.xpos[simulation.base_body_id, 2]
                viewer_text = f"  view={viewer_fps:5.1f}fps" if viewer_handle is not None else ""
                print(
                    f"t={simulation.data.time:7.2f}s  RTF={rtf:5.3f}{viewer_text}  "
                    f"base_z={base_z:6.3f}  "
                    f"EE误差={pos_error:6.3f}m/{rot_error:6.3f}rad  "
                    f"|action|max={np.max(np.abs(simulation.last_action)):6.3f}  "
                    "gripper=fixed"
                )
                if args.leg_debug:
                    joint_position, _ = simulation.joint_state()
                    leg_q = joint_position[LEG_IDS]
                    leg_actor_raw = simulation.raw_action[LEG_IDS]
                    leg_action = simulation.effective_action[LEG_IDS]
                    leg_desired = simulation.policy_desired_position[LEG_IDS]
                    leg_cmd = simulation.desired_position[LEG_IDS]
                    print(f"leg_q={format_named(LEG_NAMES, leg_q)}")
                    print(
                        "leg_actor_raw="
                        f"{format_named(LEG_NAMES, leg_actor_raw)}"
                    )
                    print(f"leg_action={format_named(LEG_NAMES, leg_action)}")
                    print(f"leg_desired={format_named(LEG_NAMES, leg_desired)}")
                    print(f"leg_cmd={format_named(LEG_NAMES, leg_cmd)}")
                    print(
                        "leg_cmd_step="
                        f"{format_named(LEG_NAMES, simulation.policy_command_step[LEG_IDS])}"
                    )
                    print(
                        "leg_track_error="
                        f"{format_named(LEG_NAMES, leg_cmd - leg_q)}"
                    )
                last_log_wall = now
                last_log_sim = simulation.data.time
                viewer_sync_count = 0

    def save_keyboard_recording() -> None:
        if (
            record_output is None
            or raw_trajectory_output is None
            or smooth_trajectory_output is None
        ):
            return
        if not record["time"]:
            print("[record] no policy samples were captured; no files written")
            return

        record_output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(values) for key, values in record.items()}
        relative_time = arrays["time"] - arrays["time"][0]
        raw_position = arrays["target_position_base"]
        raw_rotation = arrays["target_rotation_base"]
        raw_rpy = np.unwrap(
            np.stack([rpy_from_rotation(rotation) for rotation in raw_rotation]),
            axis=0,
        )
        smooth_position = gaussian_smooth(
            raw_position, args.record_smoothing, POLICY_DT
        )
        smooth_rpy = gaussian_smooth(raw_rpy, args.record_smoothing, POLICY_DT)
        smooth_rotation = np.stack(
            [rotation_from_rpy(*rpy) for rpy in smooth_rpy]
        )

        arrays.update(
            {
                "time_from_start": relative_time,
                "target_rpy_base_raw": raw_rpy,
                "target_position_base_smooth": smooth_position,
                "target_rpy_base_smooth": smooth_rpy,
                "target_rotation_base_smooth": smooth_rotation,
                "model_dir": np.array(str(model_dir.expanduser().resolve())),
                "sample_rate_hz": np.array(1.0 / POLICY_DT),
                "command_frame": np.array("base"),
                "smoothing_sigma_s": np.array(args.record_smoothing),
            }
        )
        np.savez_compressed(record_output, **arrays)

        def episode(
            position: np.ndarray,
            rotation: np.ndarray,
            trajectory_name: str,
            description: str,
        ) -> dict:
            return {
                "t": relative_time,
                "ee_pos": position,
                "ee_axis_angle": np.stack(
                    [axis_angle_from_rotation(value) for value in rotation]
                ),
                "gripper_width": arrays["gripper_command"],
                "metadata": {
                    "position_mode": "absolute_base",
                    "trajectory_name": trajectory_name,
                    "description": description,
                    "source_recording_npz": str(record_output),
                    "sample_rate_hz": 1.0 / POLICY_DT,
                    "smoothing_sigma_s": args.record_smoothing,
                },
            }

        raw_episode = episode(
            raw_position,
            raw_rotation,
            raw_trajectory_output.stem,
            "Raw target pose recorded from keyboard teleoperation",
        )
        smooth_episode = episode(
            smooth_position,
            smooth_rotation,
            smooth_trajectory_output.stem,
            "Gaussian-smoothed target pose from keyboard teleoperation",
        )
        with raw_trajectory_output.open("wb") as file:
            pickle.dump([raw_episode], file, protocol=pickle.HIGHEST_PROTOCOL)
        with smooth_trajectory_output.open("wb") as file:
            pickle.dump([smooth_episode], file, protocol=pickle.HIGHEST_PROTOCOL)
        print(
            f"[record] saved {len(relative_time)} samples "
            f"({relative_time[-1]:.2f} s) to {record_output}"
        )
        print(f"[record] raw replay trajectory: {raw_trajectory_output}")
        print(f"[record] smooth replay trajectory: {smooth_trajectory_output}")

    print(
        "[keyboard] MuJoCo 窗口或终端均可控制："
        "W/S = ±X，A/D = ±Y，R/F = ±Z；"
        "T/G = ±Roll，Y/H = ±Pitch，U/J = ±Yaw；"
        "P = 显示目标，Q = 退出；"
        f"位置步长 {args.keyboard_step:g} m，"
        f"旋转步长 {args.keyboard_rotation_step:g} rad "
        f"({math.degrees(args.keyboard_rotation_step):g} deg)"
    )
    keyboard.start()
    try:
        try:
            if args.headless:
                if run_duration <= 0 and args.freeze_after is None:
                    raise ValueError(
                        "--headless requires --duration greater than zero or --freeze-after"
                    )
                loop()
            else:
                from mujoco import viewer as mujoco_viewer

                with mujoco_viewer.launch_passive(
                    model,
                    simulation.data,
                    key_callback=keyboard.viewer_key_callback,
                ) as viewer_handle:
                    configure_viewer(
                        viewer_handle,
                        simulation.base_body_id,
                        track_robot=not args.free_camera,
                        camera_azimuth=args.camera_azimuth,
                        camera_elevation=args.camera_elevation,
                        camera_distance=args.camera_distance,
                    )
                    loop(viewer_handle)
        except KeyboardInterrupt:
            print("\n[sim2sim] 用户停止。")
    finally:
        keyboard.close()
        save_keyboard_recording()

    pos_error, rot_error = simulation.error()
    ee_position, _ = simulation.ee_pose_base()
    print(
        f"[sim2sim] 结束：EE(base)={ee_position.tolist()}, "
        f"最终误差={pos_error:.4f} m / {rot_error:.4f} rad"
    )


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[sim2sim] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
