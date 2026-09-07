#!/usr/bin/env python3

"""Play an IsaacLab task with random actions.

This is useful for checking whether joints can physically move without loading
an RL checkpoint. It intentionally bypasses the policy/runner and samples
actions directly in the task action space.
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run an IsaacLab task with random actions.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for random actions and the environment.")
parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds. Use 0 to run until closed.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.25,
    help="Uniform random action range is [-action_scale, action_scale].",
)
parser.add_argument(
    "--hold_steps",
    type=int,
    default=5,
    help="Hold each sampled action for this many environment steps.",
)
parser.add_argument(
    "--arm-only",
    action="store_true",
    help="Only randomize arm joints J1-J6; all other action dimensions stay zero.",
)
parser.add_argument(
    "--fix-base",
    action="store_true",
    help="Keep the robot root pose fixed every step so it cannot fall while the arm moves.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import ext_loco.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def find_arm_action_ids(env) -> list[int]:
    """Return action indices corresponding to J1-J6 in the joint_pos action term."""
    joint_pos_term = env.unwrapped.action_manager.get_term("joint_pos")
    joint_names = list(joint_pos_term._joint_names)
    arm_action_ids = [idx for idx, name in enumerate(joint_names) if name in {f"J{i}" for i in range(1, 7)}]
    if len(arm_action_ids) != 6:
        raise RuntimeError(
            "Expected to find J1-J6 in joint_pos action term, but got "
            f"{[(idx, joint_names[idx]) for idx in arm_action_ids]} from {joint_names}."
        )
    return arm_action_ids


def fixed_root_state(env) -> torch.Tensor:
    """Root state for pinning the robot in each environment at its current pose."""
    robot = env.unwrapped.scene["robot"]
    root_state = robot.data.root_state_w.clone()
    root_state[:, 7:] = 0.0
    return root_state


def pin_root(env, root_state: torch.Tensor) -> None:
    """Write the saved root pose/zero velocity back into the simulator."""
    robot = env.unwrapped.scene["robot"]
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    device = torch.device(env.unwrapped.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args_cli.seed)

    obs, _ = env.get_observations()
    action = torch.zeros(env.num_envs, env.num_actions, device=device)
    arm_action_ids = find_arm_action_ids(env) if args_cli.arm_only else list(range(env.num_actions))
    root_state = fixed_root_state(env) if args_cli.fix_base else None
    if root_state is not None:
        pin_root(env, root_state)
    step_dt = env.unwrapped.step_dt
    start_time = time.time()
    step_count = 0

    print(
        f"[random_play] task={args_cli.task}, num_envs={env.num_envs}, "
        f"num_actions={env.num_actions}, action_scale={args_cli.action_scale}, "
        f"arm_only={args_cli.arm_only}, fix_base={args_cli.fix_base}, "
        f"random_action_ids={arm_action_ids}"
    )

    while simulation_app.is_running():
        loop_start = time.time()
        if args_cli.duration > 0.0 and time.time() - start_time >= args_cli.duration:
            break

        with torch.inference_mode():
            if step_count % max(args_cli.hold_steps, 1) == 0:
                action.zero_()
                action[:, arm_action_ids] = (
                    2.0
                    * torch.rand(env.num_envs, len(arm_action_ids), device=device, generator=generator)
                    - 1.0
                ) * args_cli.action_scale
            obs, _, _, _ = env.step(action)
            if root_state is not None:
                pin_root(env, root_state)

        step_count += 1
        sleep_time = step_dt - (time.time() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
