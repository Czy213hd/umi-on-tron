"""Run the SF-TRON1A EEF task with zero actions and show the actual EEF frame.

This is an environment-level model check: it uses the same robot asset,
action configuration, reset logic, and EEF command term as training, but does
not load a policy checkpoint.  A zero action means joint-position actions stay
at their configured default offsets.

The Isaac Sim default timeline ends after 100 frames.  This script extends and
loops it so that the task remains open for inspection.

Run from ``IsaacLab_RFM``:

    python scripts/zero_agent.py --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-Command-Play-v0"

parser = argparse.ArgumentParser(description="Run the SF-TRON1A EEF environment with zero actions.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default: 1).")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help=f"Gym task to run (default: {DEFAULT_TASK}).")
parser.add_argument(
    "--eef-marker-scale",
    type=float,
    default=0.18,
    help="Length scale of the actual eef_link RGB axes in metres (default: 0.18).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import omni.timeline
import torch

import ext_loco.tasks  # noqa: F401  # Registers this project's Gym tasks.
from isaaclab.sim import SimulationContext
from isaaclab_tasks.utils import parse_env_cfg


def detach_stop_shutdown_callback(sim: SimulationContext) -> None:
    """Keep a transient Timeline STOP from closing this inspection session."""
    stop_handle = sim._app_control_on_stop_handle
    if stop_handle is not None:
        stop_handle.unsubscribe()
        sim._app_control_on_stop_handle = None
        print("[INFO] Detached Isaac Lab's automatic STOP-to-shutdown handler.")


def configure_inspection_timeline() -> None:
    """Prevent the default 100-frame Timeline from stopping the environment."""
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(1.0e6)
    timeline.set_looping(True)
    # Timeline edits are queued until commit(), so make them live immediately.
    timeline.commit()
    print(
        "[INFO] Timeline configured: "
        f"end={timeline.get_end_time():g}s, looping={timeline.is_looping()}."
    )


def configure_eef_check(env_cfg) -> None:
    """Make the debug marker an unambiguous view of the imported eef_link."""
    ee_command_cfg = env_cfg.commands.EE_pose
    if ee_command_cfg.body_name != "eef_link":
        raise RuntimeError(f"Expected the EEF command to track 'eef_link', got: {ee_command_cfg.body_name}")

    # The command term's current-pose marker reads the imported body's link
    # pose.  Hide its target marker so the viewport contains only the actual
    # EEF frame rather than a potentially confusing desired pose.
    ee_command_cfg.debug_vis = True
    ee_command_cfg.current_pose_visualizer_cfg.markers["frame"].scale = (args_cli.eef_marker_scale,) * 3
    ee_command_cfg.goal_pose_visualizer_cfg.markers["frame"].visible = False

    # The model update may have changed mesh files only.  Force the same URDF
    # conversion used in training so this check cannot use a stale USD cache.
    env_cfg.scene.robot.spawn.force_usd_conversion = True


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_eef_check(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    detach_stop_shutdown_callback(env.unwrapped.sim)
    configure_inspection_timeline()

    print(f"[INFO] Task: {args_cli.task}")
    print("[INFO] Zero actions are applied at every step; no policy checkpoint is loaded.")
    print("[INFO] The only visible RGB frame is the imported robot's actual eef_link frame.")
    print("[INFO] Marker prim: /Visuals/Command/body_pose")

    env.reset()
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                env.step(actions)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
