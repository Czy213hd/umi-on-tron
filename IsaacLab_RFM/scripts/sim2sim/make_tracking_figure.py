#!/usr/bin/env python3
"""Generate a paper-style MuJoCo sim2sim tracking figure and raw data.

This script reuses the production sim2sim dynamics and ONNX inference path from
``run_sf_tron1_arm_mujoco.py``.  It runs without a viewer, records the command
and measured link6 poses at policy rate, renders three representative frames,
and creates a compact multi-panel PNG/PDF figure.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from pathlib import Path

# These must be selected before importing MuJoCo/Matplotlib.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sim2sim-matplotlib")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


SCRIPT_PATH = Path(__file__).resolve()
SIM2SIM_PATH = SCRIPT_PATH.with_name("run_sf_tron1_arm_mujoco.py")
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_RUN = (
    REPO_ROOT
    / "IsaacLab_RFM/logs/rsl_rl/ImplicitOneStageARXR5Arm/2026-07-30_00-17-56"
)


def load_sim2sim_module():
    spec = importlib.util.spec_from_file_location("sf_tron1_arm_sim2sim", SIM2SIM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import sim2sim module from {SIM2SIM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model_19999 sim2sim and create a paper-style tracking figure."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_RUN / "exported")
    parser.add_argument(
        "--trajectory", type=Path, default=REPO_ROOT / "data/axis_test_xyz_world.pkl"
    )
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--trajectory-speed", type=float, default=1.0)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    parser.add_argument("--base-height", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mean-latent",
        action="store_true",
        help="Use latent mean instead of seeded sampling.",
    )
    parser.add_argument(
        "--snapshot-times",
        nargs=3,
        type=float,
        default=(3.0, 5.0, 7.0),
        metavar=("T1", "T2", "T3"),
        help="Three simulation times in seconds at which frames are rendered.",
    )
    parser.add_argument(
        "--pickup-object",
        action="store_true",
        help="Render the trajectory's pickup object and move it with the measured gripper after grasp.",
    )
    parser.add_argument(
        "--payload-gravity",
        action="store_true",
        help="Apply the trajectory payload_mass as a downward load at the gripper point.",
    )
    parser.add_argument(
        "--bottom-panel",
        choices=("error", "rpy"),
        default="error",
        help="Content of the large lower plot.",
    )
    parser.add_argument(
        "--illustrative-z-blend",
        type=float,
        default=0.0,
        help=(
            "Blend the displayed low-height Z response toward its command by this fraction "
            "(0..1). The raw measured Z remains visible and the overlay is explicitly labeled "
            "as illustrative. Raw data and metrics are never modified."
        ),
    )
    parser.add_argument(
        "--title",
        default="SF_TRON1A + ARXR5Arm sim2sim tracking — model_19999",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "IsaacLab_RFM/outputs/sim2sim_figures/model_19999_axis_tracking.png",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=REPO_ROOT / "IsaacLab_RFM/outputs/sim2sim_figures/model_19999_axis_tracking.pdf",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=REPO_ROOT / "IsaacLab_RFM/outputs/sim2sim_figures/model_19999_axis_tracking.npz",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def make_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.distance = 2.25
    camera.azimuth = 132.0
    camera.elevation = -17.0
    return camera


def render_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    base_body_id: int,
) -> np.ndarray:
    camera.lookat[:] = data.xpos[base_body_id] + np.array([0.0, 0.0, -0.15])
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def panel_label(axis, label: str) -> None:
    axis.text(
        0.018,
        0.96,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#172033",
        bbox={"boxstyle": "square,pad=0.28", "facecolor": "#cbd9f2", "edgecolor": "none"},
        zorder=20,
    )


def style_plot(axis) -> None:
    axis.grid(True, color="#d9dee8", linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.5)


def make_illustrative_positions(
    target_positions: np.ndarray,
    measured_positions: np.ndarray,
    z_blend: float,
) -> np.ndarray:
    """Return a display-only low-Z blend while preserving the raw measurements."""
    illustrative = measured_positions.copy()
    if z_blend <= 0.0:
        return illustrative
    # Fade the adjustment in only while the command is in the lower part of
    # the pickup.  smoothstep makes both joins visually continuous.
    gate = np.clip((0.15 - target_positions[:, 2]) / 0.25, 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    illustrative[:, 2] += (
        z_blend * gate * (target_positions[:, 2] - measured_positions[:, 2])
    )
    return illustrative


def create_figure(
    output: Path,
    pdf_output: Path | None,
    dpi: int,
    times: np.ndarray,
    target_positions: np.ndarray,
    measured_positions: np.ndarray,
    illustrative_positions: np.ndarray,
    illustrative_z_blend: float,
    position_errors: np.ndarray,
    orientation_errors: np.ndarray,
    target_rpy: np.ndarray,
    measured_rpy: np.ndarray,
    payload_force: np.ndarray,
    frames: list[np.ndarray],
    snapshot_times: tuple[float, float, float],
    bottom_panel: str,
    title: str,
    tracking_frame: str,
) -> None:
    colors = ("#2455ff", "#ef3b2c", "#2c8d35")
    names = ("X", "Y", "Z")
    time_from_motion = times - times[0]

    figure = plt.figure(figsize=(15.5, 8.2), constrained_layout=False, facecolor="white")
    grid = GridSpec(
        2,
        4,
        figure=figure,
        width_ratios=(1.08, 1.55, 1.55, 1.08),
        height_ratios=(1.0, 1.0),
        left=0.045,
        right=0.985,
        top=0.88,
        bottom=0.075,
        wspace=0.34,
        hspace=0.34,
    )

    snapshot_axis = figure.add_subplot(grid[0, 0])
    snapshot_axis.imshow(frames[0])
    snapshot_axis.set_axis_off()
    snapshot_axis.set_title(f"MuJoCo, $t={snapshot_times[0]:.1f}$ s", fontsize=9, pad=5)
    panel_label(snapshot_axis, "A")

    tracking_axis = figure.add_subplot(grid[0, 1:4])
    for index, (name, color) in enumerate(zip(names, colors)):
        if index == 2 and illustrative_z_blend > 0.0:
            tracking_axis.plot(
                time_from_motion,
                measured_positions[:, index],
                color="#6f747c",
                linewidth=1.0,
                linestyle=":",
                alpha=0.9,
                label="Z raw measured",
            )
            tracking_axis.plot(
                time_from_motion,
                illustrative_positions[:, index],
                color=color,
                linewidth=1.55,
                label="Z illustrative (not measured)",
            )
        else:
            tracking_axis.plot(
                time_from_motion,
                measured_positions[:, index],
                color=color,
                linewidth=1.45,
                label=f"{name} measured",
            )
        tracking_axis.plot(
            time_from_motion,
            target_positions[:, index],
            color=color,
            linewidth=1.15,
            linestyle="--",
            alpha=0.9,
            label=f"{name} command",
        )
    tracking_axis.set_xlabel("Trajectory time [s]", fontsize=10)
    tracking_axis.set_ylabel(f"{tracking_frame.capitalize()}-frame position [m]", fontsize=10)
    tracking_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.5,
    )
    style_plot(tracking_axis)
    panel_label(tracking_axis, "B")

    error_axis = figure.add_subplot(grid[1, 0:3])
    rms_cm = 100.0 * float(np.sqrt(np.mean(np.square(position_errors))))
    p95_cm = 100.0 * float(np.percentile(position_errors, 95.0))
    if bottom_panel == "rpy":
        orientation_colors = ("#7b4ab5", "#ed7d31", "#008c95")
        orientation_names = ("Roll", "Pitch", "Yaw")
        target_degrees = np.degrees(np.unwrap(target_rpy, axis=0))
        measured_degrees = np.degrees(np.unwrap(measured_rpy, axis=0))
        for index, (name, color) in enumerate(zip(orientation_names, orientation_colors)):
            error_axis.plot(
                time_from_motion,
                measured_degrees[:, index],
                color=color,
                linewidth=1.35,
                label=f"{name} measured",
            )
            error_axis.plot(
                time_from_motion,
                target_degrees[:, index],
                color=color,
                linewidth=1.1,
                linestyle="--",
                label=f"{name} command",
            )
        error_axis.set_ylabel(
            f"{tracking_frame.capitalize()}-frame orientation [deg]", fontsize=10
        )
        payload_axis = error_axis.twinx()
        payload_axis.fill_between(
            time_from_motion,
            0.0,
            payload_force,
            color="#555b66",
            alpha=0.12,
        )
        payload_axis.plot(
            time_from_motion,
            payload_force,
            color="#3d424b",
            linewidth=1.0,
            linestyle=":",
            label="Payload gravity",
        )
        payload_axis.set_ylabel("Payload gravity [N]", color="#3d424b", fontsize=10)
        payload_axis.tick_params(axis="y", labelcolor="#3d424b", labelsize=8.5)
        payload_axis.spines["top"].set_visible(False)
        error_axis.legend(
            loc="upper left",
            bbox_to_anchor=(0.065, 1.0),
            ncol=3,
            frameon=False,
            fontsize=7.8,
        )
        summary = (
            f"position RMS {rms_cm:.2f} cm  |  p95 {p95_cm:.2f} cm\n"
            f"max payload {np.max(payload_force) / 9.81:.2f} kg"
        )
    else:
        component_errors_cm = (target_positions - measured_positions) * 100.0
        for index, (name, color) in enumerate(zip(names, colors)):
            error_axis.plot(
                time_from_motion,
                component_errors_cm[:, index],
                color=color,
                linewidth=1.2,
                label=f"{name} error",
            )
        error_axis.plot(
            time_from_motion,
            position_errors * 100.0,
            color="#20242c",
            linewidth=1.6,
            label="3D error norm",
        )
        error_axis.set_ylabel("Tracking error [cm]", fontsize=10)
        error_axis.axhline(0.0, color="#7b8494", linewidth=0.75)
        error_axis.legend(loc="upper left", ncol=4, frameon=False, fontsize=8)
        axis_rms_cm = np.sqrt(np.mean(np.square(component_errors_cm), axis=0))
        summary = (
            f"3D RMS {rms_cm:.2f} cm  |  p95 {p95_cm:.2f} cm\n"
            f"axis RMS: X {axis_rms_cm[0]:.2f}, Y {axis_rms_cm[1]:.2f}, "
            f"Z {axis_rms_cm[2]:.2f} cm"
        )
    error_axis.set_xlabel("Trajectory time [s]", fontsize=10)
    style_plot(error_axis)
    error_axis.text(
        0.985,
        0.94,
        summary,
        transform=error_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c8ced9"},
    )
    panel_label(error_axis, "C")

    snapshots_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=grid[1, 3], hspace=0.16)
    for index, subgrid in enumerate(snapshots_grid):
        axis = figure.add_subplot(subgrid)
        axis.imshow(frames[index + 1])
        axis.set_axis_off()
        axis.set_title(f"$t={snapshot_times[index + 1]:.1f}$ s", fontsize=8, pad=2)
        panel_label(axis, chr(ord("D") + index))

    figure.suptitle(
        title,
        fontsize=14,
        fontweight="semibold",
        y=0.975,
    )
    if illustrative_z_blend > 0.0:
        figure.text(
            0.5,
            0.925,
            "Illustrative Z overlay — not a measured simulation signal; raw Z is retained as dotted gray",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#a12a2a",
            fontweight="semibold",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white")
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_output, facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    sim2sim = load_sim2sim_module()

    if not 0.0 <= args.illustrative_z_blend <= 1.0:
        raise ValueError("--illustrative-z-blend must be in [0, 1]")
    if args.physics_dt <= 0.0 or not np.isfinite(args.physics_dt):
        raise ValueError("--physics-dt must be finite and positive")
    policy_decimation = round(sim2sim.POLICY_DT / args.physics_dt)
    if not np.isclose(policy_decimation * args.physics_dt, sim2sim.POLICY_DT):
        raise ValueError("--physics-dt must divide the 0.02 s policy period exactly")

    trajectory = sim2sim.PickleTrajectory(
        args.trajectory,
        args.trajectory_index,
        args.start_delay,
        False,
        True,
        args.trajectory_speed,
    )
    with args.trajectory.expanduser().resolve().open("rb") as file:
        episodes = pickle.load(file)
    if isinstance(episodes, dict) and "episodes" in episodes:
        episodes = episodes["episodes"]
    episode = episodes[args.trajectory_index % len(episodes)]
    trajectory_payload_mass = np.asarray(
        episode.get("payload_mass", np.zeros(len(trajectory.positions))), dtype=np.float64
    )
    if trajectory_payload_mass.shape != (len(trajectory.positions),):
        raise ValueError("trajectory payload_mass must have one value per pose frame")
    trajectory_metadata = episode.get("metadata", {}) or {}
    object_position_world = np.asarray(
        trajectory_metadata.get("object_position_world", [0.64, 0.0, 0.06]),
        dtype=np.float64,
    )
    pickup_tip_offset = np.asarray(
        trajectory_metadata.get("pickup_tip_offset_link6", [0.15, 0.0, 0.0]),
        dtype=np.float64,
    )
    grasp_time = float(trajectory_metadata.get("grasp_time_s", np.inf))
    model = sim2sim.load_sim_model(
        sim2sim.DEFAULT_MJCF,
        physics_dt=args.physics_dt,
        add_door=False,
        hide_current_ee_frame=False,
        add_pickup_object=args.pickup_object,
    )
    policy = sim2sim.ThreeOnnxPolicy(
        args.model_dir,
        sample_latent=not args.mean_latent,
        rng=np.random.default_rng(args.seed),
    )
    initial_command_frame = (
        "base" if trajectory.position_mode == "absolute_base" else "world"
    )
    initial_command = np.concatenate(
        (
            trajectory.positions[0],
            sim2sim.rpy_from_rotation(trajectory.rotations[0]),
        )
    )
    simulation = sim2sim.Sim2Sim(
        model,
        policy,
        initial_command,
        initial_command_frame,
        args.base_height,
        0.0,
        0.0,
        False,
        0.0,
        policy_decimation,
        1.0,
        sim2sim.ACTION_SCALE,
    )

    pickup_mocap_id = -1
    if args.pickup_object:
        pickup_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "sim2sim_pickup_object"
        )
        if pickup_body_id < 0:
            raise ValueError("pickup object was requested but is missing from the model")
        pickup_mocap_id = int(model.body_mocapid[pickup_body_id])
        simulation.data.mocap_pos[pickup_mocap_id] = object_position_world
        simulation.data.mocap_quat[pickup_mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, simulation.data)

    # The source MJCF keeps MuJoCo's default 640x480 offscreen framebuffer.
    # Render at that native size and let the final high-DPI figure handle scale.
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = make_camera(model)
    snapshot_times = tuple(float(value) for value in args.snapshot_times)
    frames: list[np.ndarray] = []
    next_snapshot = 0

    times: list[float] = []
    targets: list[np.ndarray] = []
    measured: list[np.ndarray] = []
    targets_world: list[np.ndarray] = []
    measured_world: list[np.ndarray] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    target_rpy: list[np.ndarray] = []
    measured_rpy: list[np.ndarray] = []
    payload_forces: list[float] = []
    base_positions: list[np.ndarray] = []
    joint_positions: list[np.ndarray] = []
    joint_torques: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []

    run_duration = trajectory.start_delay + trajectory.duration + sim2sim.POLICY_DT
    max_steps = int(np.ceil(run_duration / args.physics_dt))
    for physics_step in range(max_steps):
        elapsed = max(0.0, simulation.data.time - trajectory.start_delay)
        source_time = min(
            elapsed * trajectory.playback_speed,
            trajectory.source_duration - trajectory.sample_dt,
        )
        source_frame = min(
            int(source_time / trajectory.sample_dt),
            len(trajectory_payload_mass) - 1,
        )
        payload_mass = float(trajectory_payload_mass[source_frame])
        policy_updated = physics_step % policy_decimation == 0
        if policy_updated:
            trajectory.update(simulation)

        simulation.data.qfrc_applied[:] = 0.0
        ee_position_before = simulation.data.xpos[simulation.ee_body_id].copy()
        ee_rotation_before = simulation.data.xmat[simulation.ee_body_id].reshape(3, 3).copy()
        pickup_point_world = ee_position_before + ee_rotation_before @ pickup_tip_offset
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
            if source_time >= grasp_time:
                simulation.data.mocap_pos[pickup_mocap_id] = pickup_point_world
                pickup_quaternion = np.empty(4, dtype=np.float64)
                mujoco.mju_mat2Quat(pickup_quaternion, ee_rotation_before.reshape(-1))
                simulation.data.mocap_quat[pickup_mocap_id] = pickup_quaternion
            else:
                simulation.data.mocap_pos[pickup_mocap_id] = object_position_world
                simulation.data.mocap_quat[pickup_mocap_id] = [1.0, 0.0, 0.0, 0.0]
        simulation.step(physics_step)

        if policy_updated:
            target_position_world, target_rotation_world = simulation.target_pose_world()
            ee_position_world = simulation.data.xpos[simulation.ee_body_id].copy()
            ee_rotation_world = simulation.data.xmat[simulation.ee_body_id].reshape(3, 3).copy()
            if trajectory.position_mode == "absolute_base":
                target_position, target_rotation = simulation.target_pose_base()
                ee_position, ee_rotation = simulation.ee_pose_base()
            else:
                target_position, target_rotation = target_position_world, target_rotation_world
                ee_position, ee_rotation = ee_position_world, ee_rotation_world
            times.append(float(simulation.data.time))
            targets.append(target_position.copy())
            measured.append(ee_position)
            targets_world.append(target_position_world.copy())
            measured_world.append(ee_position_world.copy())
            position_errors.append(float(np.linalg.norm(target_position - ee_position)))
            orientation_errors.append(
                sim2sim.rotation_angle(target_rotation @ ee_rotation.T)
            )
            target_rpy.append(sim2sim.rpy_from_rotation(target_rotation))
            measured_rpy.append(sim2sim.rpy_from_rotation(ee_rotation))
            payload_forces.append(payload_mass * 9.81 if args.payload_gravity else 0.0)
            base_positions.append(simulation.data.xpos[simulation.base_body_id].copy())
            joint_position, _ = simulation.joint_state()
            joint_positions.append(joint_position)
            joint_torques.append(simulation.last_torque.copy())
            raw_actions.append(simulation.raw_action.copy())

        while (
            next_snapshot < len(snapshot_times)
            and simulation.data.time >= snapshot_times[next_snapshot]
        ):
            frames.append(
                render_frame(renderer, simulation.data, camera, simulation.base_body_id)
            )
            next_snapshot += 1

    while len(frames) < 3:
        frames.append(render_frame(renderer, simulation.data, camera, simulation.base_body_id))
    renderer.close()

    time_array = np.asarray(times)
    target_array = np.asarray(targets)
    measured_array = np.asarray(measured)
    illustrative_array = make_illustrative_positions(
        target_array, measured_array, args.illustrative_z_blend
    )
    position_error_array = np.asarray(position_errors)
    orientation_error_array = np.asarray(orientation_errors)
    target_rpy_array = np.asarray(target_rpy)
    measured_rpy_array = np.asarray(measured_rpy)
    payload_force_array = np.asarray(payload_forces)

    # Remove the stabilization prefix so the plots report trajectory tracking only.
    active = time_array >= trajectory.start_delay
    plot_times = time_array[active]
    plot_targets = target_array[active]
    plot_measured = measured_array[active]
    plot_illustrative = illustrative_array[active]
    plot_position_errors = position_error_array[active]
    plot_orientation_errors = orientation_error_array[active]
    plot_target_rpy = target_rpy_array[active]
    plot_measured_rpy = measured_rpy_array[active]
    plot_payload_force = payload_force_array[active]

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.data_output,
        checkpoint=np.array(str(DEFAULT_RUN / "model_19999.pt")),
        model_dir=np.array(str(args.model_dir.expanduser().resolve())),
        trajectory=np.array(str(args.trajectory.expanduser().resolve())),
        tracking_frame=np.array(
            "base" if trajectory.position_mode == "absolute_base" else "world"
        ),
        time=time_array,
        target_position=target_array,
        measured_position=measured_array,
        illustrative_position=illustrative_array,
        illustrative_z_blend=np.array(args.illustrative_z_blend),
        target_position_world=np.asarray(targets_world),
        measured_position_world=np.asarray(measured_world),
        position_error=position_error_array,
        orientation_error=orientation_error_array,
        target_rpy_world=target_rpy_array,
        measured_rpy_world=measured_rpy_array,
        payload_force=payload_force_array,
        base_position_world=np.asarray(base_positions),
        joint_position=np.asarray(joint_positions),
        joint_torque=np.asarray(joint_torques),
        raw_action=np.asarray(raw_actions),
        snapshot_times=np.asarray(snapshot_times),
        snapshot_frames=np.asarray(frames),
    )

    create_figure(
        args.output,
        args.pdf,
        args.dpi,
        plot_times,
        plot_targets,
        plot_measured,
        plot_illustrative,
        args.illustrative_z_blend,
        plot_position_errors,
        plot_orientation_errors,
        plot_target_rpy,
        plot_measured_rpy,
        plot_payload_force,
        frames,
        snapshot_times,
        args.bottom_panel,
        args.title,
        "base" if trajectory.position_mode == "absolute_base" else "world",
    )
    print(f"[figure] PNG: {args.output.expanduser().resolve()}")
    print(f"[figure] PDF: {args.pdf.expanduser().resolve()}")
    print(f"[figure] data: {args.data_output.expanduser().resolve()}")
    print(
        "[figure] position RMS="
        f"{100.0 * np.sqrt(np.mean(np.square(plot_position_errors))):.3f} cm, "
        f"p95={100.0 * np.percentile(plot_position_errors, 95.0):.3f} cm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
