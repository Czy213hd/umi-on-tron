#!/usr/bin/env python3
"""Run the SF_TRON1A + ARXR5 arm policy in MuJoCo.

The inference chain matches the IsaacLab/RSL-RL export:

    10 x 55 contact observations -> contactNet -> GRU -> 67-D latent
    65-D policy observation + 67-D latent -> actor -> 14 joint targets

If ``--command`` is omitted, the program asks for a final end-effector pose
as ``x y z roll pitch yaw`` before opening the viewer.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import threading
import time
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
DEFAULT_TRAJECTORY = Path("/home/phi5090ii/UMI-ON-TRON/data/pushing.pkl")
TIP_OFFSET_POS = np.array([0.08657, -0.0249, -0.00024366], dtype=np.float64)
TIP_OFFSET_RPY = (-math.pi * 0.5, 0.0, -math.pi * 0.5)

# This order must match PointfootCfg.init_state.joint_names and the training
# articulation order. It is intentionally not MuJoCo's internal joint order.
JOINT_NAMES = (
    "J1",
    "abad_L_Joint",
    "abad_R_Joint",
    "J2",
    "hip_L_Joint",
    "hip_R_Joint",
    "J3",
    "knee_L_Joint",
    "knee_R_Joint",
    "J4",
    "ankle_L_Joint",
    "ankle_R_Joint",
    "J5",
    "J6",
)
DEFAULT_JOINT_POS = np.array(
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

# IsaacLab actuator gains used by LIMX_SF_TRON1A_ARM.
KP = np.array(
    [18.0, 40.0, 40.0, 18.0, 40.0, 40.0, 18.0, 40.0, 40.0, 4.0, 45.0, 45.0, 4.0, 4.0],
    dtype=np.float64,
)
KD = np.array(
    [1.0, 1.8, 1.8, 1.0, 1.8, 1.8, 1.0, 1.8, 1.8, 0.5, 0.8, 0.8, 0.5, 0.5],
    dtype=np.float64,
)
TORQUE_LIMIT = np.array(
    [18.0, 80.0, 80.0, 18.0, 80.0, 80.0, 18.0, 80.0, 80.0, 3.0, 40.0, 40.0, 3.0, 3.0],
    dtype=np.float64,
)

# MuJoCo needs a finer contact step than the source PhysX simulation to keep
# the detailed ankle meshes from visibly tunnelling into the floor. The policy
# frequency remains identical to training: 0.001 * 20 = 0.02 s (50 Hz).
PHYSICS_DT = 0.001
POLICY_DECIMATION = 20
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
HISTORY_LENGTH = 10
OBS_DIM = 65
CONTACT_OBS_DIM = 55
ACTION_DIM = 14


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
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing actor.onnx, contactNet.onnx and gru.onnx. "
        "Default: newest exported training run.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Simulation duration in seconds; 0 runs until the viewer closes. "
        "Default: 30 s manually, or one complete trajectory.",
    )
    parser.add_argument("--base-height", type=float, default=0.84)
    parser.add_argument(
        "--sample-latent",
        action="store_true",
        help="Sample the predicted latent distribution instead of using its mean.",
    )
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
        "--keyboard-step",
        type=float,
        default=0.02,
        help="Target-position increment per key press in metres (default: 0.02).",
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
        "--no-planar-center",
        action="store_true",
        help="Disable the XY centering used by IsaacLab PicklePoseSequenceCommand.",
    )
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_args()


def newest_exported_model_dir() -> Path:
    log_root = ISAACLAB_ROOT / "logs/rsl_rl/ImplicitOneStageARXR5Arm"
    candidates = [
        path
        for path in log_root.glob("*/exported")
        if all((path / name).is_file() for name in ("actor.onnx", "contactNet.onnx", "gru.onnx"))
    ]
    if candidates:
        return max(candidates, key=lambda path: path.parent.name)
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


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(float(cosine)))


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
        self.positions = np.asarray(episode["ee_pos"], dtype=np.float64).copy()
        axis_angles = np.asarray(episode["ee_axis_angle"], dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"Unexpected ee_pos shape: {self.positions.shape}")
        if axis_angles.shape != self.positions.shape:
            raise ValueError(
                f"ee_axis_angle shape {axis_angles.shape} does not match ee_pos {self.positions.shape}"
            )
        if planar_center:
            if len(self.positions) < 4:
                raise ValueError("planar_center requires at least four trajectory frames")
            self.positions[:, :2] -= self.positions[1:4, :2].mean(axis=0)

        self.rotations = np.stack([rotation_from_axis_angle(value) for value in axis_angles])
        time_samples = np.asarray(episode.get("t", []), dtype=np.float64)
        if len(time_samples) >= 2:
            self.sample_dt = float(np.median(np.diff(time_samples)))
        else:
            self.sample_dt = 0.005
        if self.sample_dt <= 0:
            raise ValueError(f"Invalid trajectory sample dt: {self.sample_dt}")

        tip_rotation = rotation_from_rpy(*TIP_OFFSET_RPY)
        self.tip_rotation_inverse = tip_rotation.T
        self.tip_position_inverse = -self.tip_rotation_inverse @ TIP_OFFSET_POS
        self.start_delay = max(0.0, float(start_delay))
        self.loop = loop
        self.duration = len(self.positions) * self.sample_dt
        self.world_offset = np.zeros(3, dtype=np.float64)
        self.command_origin: np.ndarray | None = None
        self.current_cycle = -1
        self.finished_message_printed = False

    def translate_offset(self, delta: np.ndarray) -> None:
        self.world_offset += np.asarray(delta, dtype=np.float64)

    def update(self, simulation: "Sim2Sim") -> None:
        elapsed = simulation.data.time - self.start_delay
        if elapsed < 0:
            return

        if self.loop:
            cycle = int(elapsed // self.duration)
            local_time = elapsed - cycle * self.duration
        else:
            cycle = 0
            local_time = min(elapsed, self.duration - self.sample_dt)

        new_cycle = self.command_origin is None or cycle != self.current_cycle
        if new_cycle:
            self.command_origin = simulation.data.xpos[simulation.ee_body_id].copy()
            self.current_cycle = cycle
            self.finished_message_printed = False
            print(
                f"[trajectory] episode={self.episode_index}, cycle={cycle}, "
                f"origin={self.command_origin.round(4).tolist()}"
            )

        frame = min(int(local_time / self.sample_dt), len(self.positions) - 1)
        tip_position = self.positions[frame]
        tip_rotation = self.rotations[frame]
        link_position = tip_position + tip_rotation @ self.tip_position_inverse
        link_rotation = tip_rotation @ self.tip_rotation_inverse
        world_position = self.command_origin + link_position + self.world_offset
        simulation.set_target(world_position, link_rotation, reset_reference=new_cycle)

        if not self.loop and elapsed >= self.duration and not self.finished_message_printed:
            print("[trajectory] playback complete; holding the final pose.")
            self.finished_message_printed = True


def load_sim_model(mjcf_path: Path) -> mujoco.MjModel:
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
    option.set("timestep", str(PHYSICS_DT))

    # The MuJoCo defaults (solref=0.02 1) are visibly too soft for these
    # high-resolution foot collision meshes. Keep the contact stable and firm
    # without making it perfectly rigid.
    collision_default = root.find("./default/default[@class='collision']/geom")
    if collision_default is not None:
        collision_default.set("solref", "0.005 1")
        collision_default.set("solimp", "0.95 0.99 0.001")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            {"diffuse": "0.7 0.7 0.7", "ambient": "0.25 0.25 0.25", "specular": "0.2 0.2 0.2"},
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
                "rgba": "0.32 0.35 0.38 1",
                "friction": "0.8 0.6 0.001",
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
            {"name": "sim2sim_light", "pos": "0 -1 3", "dir": "0 0 -1", "directional": "true"},
        ),
    )
    worldbody.insert(
        2,
        ET.Element(
            "site",
            {
                "name": "command_target",
                "type": "sphere",
                "pos": "0.15 0 1",
                "size": "0.035",
                "rgba": "1 0.1 0.1 0.8",
                "group": "0",
            },
        ),
    )
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
        return np.clip(action, -100.0, 100.0)


class KeyboardTargetController:
    """Thread-safe target deltas produced by the MuJoCo viewer callback."""

    KEY_DELTAS = {
        ord("W"): np.array([1.0, 0.0, 0.0]),
        ord("S"): np.array([-1.0, 0.0, 0.0]),
        ord("A"): np.array([0.0, 1.0, 0.0]),
        ord("D"): np.array([0.0, -1.0, 0.0]),
        ord("R"): np.array([0.0, 0.0, 1.0]),
        ord("F"): np.array([0.0, 0.0, -1.0]),
    }

    def __init__(self, step: float):
        if step <= 0:
            raise ValueError("--keyboard-step must be greater than zero")
        self.step = float(step)
        self._pending_delta = np.zeros(3, dtype=np.float64)
        self._print_requested = False
        self._lock = threading.Lock()

    def callback(self, keycode: int) -> None:
        with self._lock:
            direction = self.KEY_DELTAS.get(keycode)
            if direction is not None:
                self._pending_delta += direction * self.step
            elif keycode == ord("P"):
                self._print_requested = True

    def consume(self) -> tuple[np.ndarray, bool]:
        with self._lock:
            delta = self._pending_delta.copy()
            print_requested = self._print_requested
            self._pending_delta.fill(0.0)
            self._print_requested = False
        return delta, print_requested


class Sim2Sim:
    def __init__(
        self,
        model: mujoco.MjModel,
        policy: ThreeOnnxPolicy,
        command: np.ndarray,
        command_frame: str,
        base_height: float,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.policy = policy
        self.command_frame = command_frame
        self.target_position = command[:3].copy()
        self.target_rotation = rotation_from_rpy(*command[3:])

        self.joint_qpos_adr = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=int)
        self.joint_dof_adr = np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES], dtype=int)
        self.motor_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_motor") for name in JOINT_NAMES],
            dtype=int,
        )
        if np.any(self.motor_ids < 0):
            raise ValueError("One or more joint motors are missing from the MJCF")

        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_Link")
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        self.target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "command_target")
        self.imu_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if min(self.base_body_id, self.ee_body_id, self.target_site_id, self.imu_sensor_id) < 0:
            raise ValueError("Required base/EE/IMU/target elements are missing")

        mujoco.mj_resetData(model, self.data)
        root_adr = model.joint("root").qposadr[0]
        self.data.qpos[root_adr : root_adr + 7] = [0.0, 0.0, base_height, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.joint_qpos_adr] = DEFAULT_JOINT_POS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(model, self.data)

        self.raw_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self.last_torque = np.zeros(ACTION_DIM, dtype=np.float64)
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
        no_ankle = np.array(["ankle" not in name for name in JOINT_NAMES])
        observation = np.concatenate(
            (
                self.base_angular_velocity(),
                self.projected_gravity(),
                (position - DEFAULT_JOINT_POS)[no_ankle],
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
        no_ankle = np.array(["ankle" not in name for name in JOINT_NAMES])
        observation = np.concatenate(
            (
                self.base_angular_velocity(),
                self.projected_gravity(),
                self.pose_6d(target_position, target_rotation),
                (position - DEFAULT_JOINT_POS)[no_ankle],
                velocity,
                self.last_action,
                self.pose_6d(ee_position, ee_rotation),
                np.array([self.se3_distance_reference]),
            )
        )
        if observation.shape != (OBS_DIM,):
            raise RuntimeError(f"Policy observation has shape {observation.shape}")
        return np.clip(observation, -100.0, 100.0)

    def _initial_se3_distance(self) -> float:
        ee_position, ee_rotation = self.ee_pose_base()
        target_position, target_rotation = self.target_pose_base()
        return float(
            2.0 * np.linalg.norm(target_position - ee_position)
            + rotation_angle(target_rotation @ ee_rotation.T)
        )

    def _update_target_marker(self) -> None:
        target_position, _ = self.target_pose_world()
        self.model.site_pos[self.target_site_id] = target_position

    def translate_target(self, delta: np.ndarray) -> None:
        """Translate the target in the selected command frame."""
        self.target_position += np.asarray(delta, dtype=np.float64)
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

    def infer(self) -> None:
        contact_obs = self.contact_observation()
        self.history.append(contact_obs)
        history = np.stack(self.history, axis=0)
        self.raw_action = self.policy(self.policy_observation(), history)
        self.se3_distance_reference = max(0.0, self.se3_distance_reference - POLICY_DT)

    def apply_pd(self) -> None:
        position, velocity = self.joint_state()
        # Same torque-aware action clamp used by SolefootController.cpp.
        action_min = position - DEFAULT_JOINT_POS + (KD * velocity - TORQUE_LIMIT) / KP
        action_max = position - DEFAULT_JOINT_POS + (KD * velocity + TORQUE_LIMIT) / KP
        effective_action = np.clip(self.raw_action, action_min, action_max)
        desired_position = DEFAULT_JOINT_POS + effective_action
        torque = KP * (desired_position - position) - KD * velocity
        torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.motor_ids] = torque
        self.last_action = effective_action
        self.last_torque = torque

    def step(self, physics_step: int) -> None:
        if physics_step % POLICY_DECIMATION == 0:
            self.infer()
        self.apply_pd()
        mujoco.mj_step(self.model, self.data)
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


def configure_viewer(viewer_handle, base_body_id: int, track_robot: bool) -> None:
    if track_robot:
        viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer_handle.cam.trackbodyid = base_body_id
        viewer_handle.cam.fixedcamid = -1
    else:
        viewer_handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer_handle.cam.trackbodyid = -1
    viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.45]
    viewer_handle.cam.distance = 2.4
    viewer_handle.cam.azimuth = 135.0
    viewer_handle.cam.elevation = -18.0
    viewer_handle.opt.geomgroup[2] = 1
    viewer_handle.opt.geomgroup[3] = 0


def run(args: argparse.Namespace) -> None:
    trajectory = None
    if args.trajectory is not None:
        trajectory = PickleTrajectory(
            args.trajectory,
            args.trajectory_index,
            args.trajectory_start_delay,
            args.trajectory_loop,
            planar_center=not args.no_planar_center,
        )
        # This is the fallback hold target before trajectory anchoring starts.
        command = np.asarray(
            args.command if args.command is not None else [0.15, 0.0, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        command_frame = "world"
    else:
        command = read_command(args.command)
        command_frame = args.command_frame

    model_dir = args.model_dir or newest_exported_model_dir()
    rng = np.random.default_rng(args.seed)

    print(f"[sim2sim] MJCF: {args.mjcf.expanduser().resolve()}")
    print(f"[sim2sim] ONNX: {model_dir.expanduser().resolve()}")
    print(
        "[sim2sim] 最终点 "
        f"({command_frame}): xyz={command[:3].tolist()}, "
        f"rpy={command[3:].tolist()} rad"
    )
    if trajectory is not None:
        print(
            f"[trajectory] file={trajectory.path}, episode={trajectory.episode_index}, "
            f"frames={len(trajectory.positions)}, sample_dt={trajectory.sample_dt:g}s, "
            f"duration={trajectory.duration:.3f}s, start_delay={trajectory.start_delay:g}s"
        )

    model = load_sim_model(args.mjcf)
    policy = ThreeOnnxPolicy(model_dir, args.sample_latent, rng)
    simulation = Sim2Sim(model, policy, command, command_frame, args.base_height)
    keyboard = KeyboardTargetController(args.keyboard_step)

    if args.duration is None:
        if trajectory is None:
            run_duration = 30.0
        elif trajectory.loop:
            run_duration = 0.0
        else:
            # Include one more policy tick so the final recorded frame is
            # applied before an automatic non-looping run exits.
            run_duration = trajectory.start_delay + trajectory.duration + POLICY_DT
    else:
        run_duration = args.duration
    max_steps = math.inf if run_duration <= 0 else math.ceil(run_duration / PHYSICS_DT)
    log_steps = max(1, round(args.log_interval / PHYSICS_DT))
    render_steps = max(1, round(1.0 / (max(args.render_fps, 1.0) * PHYSICS_DT)))
    wall_start = time.perf_counter()

    def loop(viewer_handle=None) -> None:
        physics_step = 0
        while physics_step < max_steps and (viewer_handle is None or viewer_handle.is_running()):
            target_delta, print_target = keyboard.consume()
            if np.any(target_delta):
                if trajectory is not None:
                    trajectory.translate_offset(target_delta)
                    print(
                        "[keyboard] trajectory world offset="
                        f"{trajectory.world_offset.round(4).tolist()}"
                    )
                else:
                    simulation.translate_target(target_delta)
                    print(
                        f"[keyboard] target({command_frame})="
                        f"{simulation.target_position.round(4).tolist()}"
                    )
            elif print_target:
                print(
                    f"[keyboard] target({command_frame})="
                    f"{simulation.target_position.round(4).tolist()}"
                )
            if trajectory is not None and physics_step % POLICY_DECIMATION == 0:
                trajectory.update(simulation)
            simulation.step(physics_step)
            physics_step += 1
            # Physics runs at 1 kHz, but synchronizing the GUI at 1 kHz makes
            # wall-clock time lag badly and looks like slow motion.
            if viewer_handle is not None and physics_step % render_steps == 0:
                viewer_handle.sync()
            if physics_step % log_steps == 0:
                pos_error, rot_error = simulation.error()
                base_z = simulation.data.xpos[simulation.base_body_id, 2]
                print(
                    f"t={simulation.data.time:7.2f}s  base_z={base_z:6.3f}  "
                    f"EE误差={pos_error:6.3f}m/{rot_error:6.3f}rad  "
                    f"|action|max={np.max(np.abs(simulation.last_action)):6.3f}"
                )
            if not args.no_realtime:
                deadline = wall_start + simulation.data.time
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

    try:
        if args.headless:
            if run_duration <= 0:
                raise ValueError("--headless requires --duration greater than zero")
            loop()
        else:
            import mujoco.viewer

            print(
                "[keyboard] 移动目标点：W/S = ±X，A/D = ±Y，R/F = ±Z，"
                f"P = 显示坐标；步长 {args.keyboard_step:g} m"
            )
            with mujoco.viewer.launch_passive(
                model, simulation.data, key_callback=keyboard.callback
            ) as viewer_handle:
                configure_viewer(
                    viewer_handle,
                    simulation.base_body_id,
                    track_robot=not args.free_camera,
                )
                loop(viewer_handle)
    except KeyboardInterrupt:
        print("\n[sim2sim] 用户停止。")

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
