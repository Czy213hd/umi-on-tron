#!/usr/bin/env python3
"""Deterministic multi-environment evaluation for the four architecture/reward ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--variant",
    required=True,
    choices=("transformer_gru", "gru_only", "transformer_only", "no_latent", "cenet", "privileged_oracle"),
)
parser.add_argument("--model-name", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--num-targets", type=int, default=1)
parser.add_argument("--target-offset", type=int, default=0)
parser.add_argument("--target-population", type=int, default=None)
parser.add_argument("--target-duration", type=float, default=8.0)
parser.add_argument("--final-window", type=float, default=1.0)
parser.add_argument("--hold-duration", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--x-min", type=float, default=-0.3)
parser.add_argument("--x-max", type=float, default=0.8)
parser.add_argument("--z-min", type=float, default=0.4)
parser.add_argument("--z-max", type=float, default=1.7)
parser.add_argument("--rpy-min", type=float, default=-0.3)
parser.add_argument("--rpy-max", type=float, default=0.3)
parser.add_argument("--task", default="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0")
parser.add_argument("--asset-usd-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import hashlib
import json
import math
import os
import time

import gymnasium as gym
import numpy as np
import torch

import ext_loco.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import ImplicitOneStageRunner


FIELDNAMES = (
    "model",
    "variant",
    "checkpoint",
    "seed",
    "env_id",
    "target_id",
    "target_x",
    "target_y",
    "target_z",
    "target_roll",
    "target_pitch",
    "target_yaw",
    "final_pos_mean_m",
    "final_pos_p95_m",
    "final_ori_mean_rad",
    "final_ori_p95_rad",
    "success_5cm_5deg",
    "final_strict_stable_success_3cm_5deg",
    "final_relaxed_stable_success_5cm_7deg",
    "final_strict_in_threshold_ratio",
    "final_relaxed_in_threshold_ratio",
    "terminated",
    "termination_reason",
    "action_rate_l2_mean",
    "action_rms",
)


def build_targets(num_targets: int, num_envs: int, seed: int) -> np.ndarray:
    """Return reproducible relative XY / absolute Z / RPY targets."""
    rng = np.random.default_rng(seed)
    low = np.asarray(
        (args_cli.x_min, -0.5, args_cli.z_min, args_cli.rpy_min, args_cli.rpy_min, args_cli.rpy_min),
        dtype=np.float32,
    )
    high = np.asarray(
        (args_cli.x_max, 0.5, args_cli.z_max, args_cli.rpy_max, args_cli.rpy_max, args_cli.rpy_max),
        dtype=np.float32,
    )
    return rng.uniform(low, high, size=(num_targets, num_envs, 6)).astype(np.float32)


def target_hash(targets: np.ndarray) -> str:
    return hashlib.sha256(targets.tobytes()).hexdigest()


def configure_agent(agent_cfg, variant: str) -> None:
    if variant == "no_latent":
        agent_cfg.ppo_algorithm.use_latent = False
    elif variant == "privileged_oracle":
        agent_cfg.ppo_algorithm.use_latent = False
        agent_cfg.ppo_algorithm.use_privileged_actor = True
    elif variant == "cenet":
        agent_cfg.ppo_algorithm.use_latent = True
        agent_cfg.contactNet.class_name = "CENet"
        agent_cfg.gru.class_name = "IdentityGRUWrapper"
    elif variant == "gru_only":
        agent_cfg.contactNet.class_name = "LastObservationEncoder"
        agent_cfg.gru.class_name = "GRUWrapper"
    elif variant == "transformer_only":
        agent_cfg.contactNet.class_name = "SimplifiedContactNetModel"
        agent_cfg.gru.class_name = "IdentityGRUWrapper"


def deterministic_action(
    runner,
    obs: torch.Tensor,
    critic_obs: torch.Tensor,
    history: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    normalized_obs = runner.obs_normalizer(obs)
    if variant == "privileged_oracle":
        normalized_critic_obs = runner.critic_obs_normalizer(critic_obs)
        return runner.actor_critic.act_inference(normalized_critic_obs)
    if variant == "no_latent":
        return runner.actor_critic.act_inference(normalized_obs)

    normalized_history = runner.contactNet_obs_normalizer(history)
    encoder_output = runner.contactNet(normalized_history)
    if variant in ("transformer_gru", "gru_only"):
        encoded = runner.gru.gru_forward(encoder_output, runner.gru.hidden_state)
    else:
        encoded = encoder_output
    latent_dim = runner.ppo_alg.next_obs_latent_dim
    velocity = encoded[:, :3]
    latent_mean = encoded[:, 3 : 3 + latent_dim]
    actor_input = torch.cat((normalized_obs, velocity, latent_mean), dim=-1)
    return runner.actor_critic.act_inference(actor_input)


def install_target(command_term, target_values: np.ndarray) -> torch.Tensor:
    device = command_term.device
    values = torch.as_tensor(target_values, device=device)
    root_position = command_term.robot.data.root_link_pos_w
    pose = torch.empty((command_term.num_envs, 7), device=device, dtype=root_position.dtype)
    pose[:, 0] = root_position[:, 0] + values[:, 0]
    pose[:, 1] = root_position[:, 1] + values[:, 1]
    pose[:, 2] = values[:, 2]
    quaternion = quat_from_euler_xyz(values[:, 3], values[:, 4], values[:, 5])
    pose[:, 3:] = quat_unique(quaternion)
    command_term.pose_command_w.copy_(pose)
    command_term.time_left.fill_(1.0e6)
    command_term._set_travel_direction(torch.arange(command_term.num_envs, device=device))
    command_term._update_se3_ref(torch.arange(command_term.num_envs, device=device))
    return pose


def summarize_rows(rows: list[dict], targets: np.ndarray) -> dict[str, float | int | str]:
    pos = np.asarray([row["final_pos_mean_m"] for row in rows], dtype=np.float64)
    ori = np.asarray([row["final_ori_mean_rad"] for row in rows], dtype=np.float64)
    success = np.asarray([row["success_5cm_5deg"] for row in rows], dtype=np.float64)
    strict_success = np.asarray(
        [row["final_strict_stable_success_3cm_5deg"] for row in rows], dtype=np.float64
    )
    relaxed_success = np.asarray(
        [row["final_relaxed_stable_success_5cm_7deg"] for row in rows], dtype=np.float64
    )
    terminated = np.asarray([row["terminated"] for row in rows], dtype=np.float64)
    action_rate = np.asarray([row["action_rate_l2_mean"] for row in rows], dtype=np.float64)
    action_rms = np.asarray([row["action_rms"] for row in rows], dtype=np.float64)
    finite_pos = pos[np.isfinite(pos)]
    finite_ori = ori[np.isfinite(ori)]
    return {
        "model": args_cli.model_name,
        "variant": args_cli.variant,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "target_hash": target_hash(targets),
        "num_commands": len(rows),
        "position_mean_m": float(np.mean(finite_pos)),
        "position_median_m": float(np.median(finite_pos)),
        "position_p95_m": float(np.quantile(finite_pos, 0.95)),
        "orientation_mean_rad": float(np.mean(finite_ori)),
        "orientation_median_rad": float(np.median(finite_ori)),
        "orientation_p95_rad": float(np.quantile(finite_ori, 0.95)),
        "success_5cm_5deg_rate": float(np.mean(success)),
        "final_strict_stable_success_3cm_5deg_rate": float(np.mean(strict_success)),
        "final_relaxed_stable_success_5cm_7deg_rate": float(np.mean(relaxed_success)),
        "termination_rate": float(np.mean(terminated)),
        "action_rate_l2_mean": float(np.mean(action_rate)),
        "action_rate_l2_p95": float(np.quantile(action_rate, 0.95)),
        "action_rms_mean": float(np.mean(action_rms)),
        "peak_cuda_memory_mib": float(torch.cuda.max_memory_allocated() / 2**20),
    }


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.num_targets <= 0:
        raise ValueError("--num-envs and --num-targets must be positive")
    target_population = args_cli.target_population or args_cli.num_envs
    if args_cli.target_offset < 0 or args_cli.target_offset + args_cli.num_envs > target_population:
        raise ValueError("target offset/range must fit inside --target-population")
    if args_cli.num_targets != 1:
        raise ValueError(
            "This Isaac Sim 4.5 environment exits when its command is replaced a second time; "
            "use --num-targets 1 and increase --num-envs for independent random samples."
        )
    if args_cli.target_duration <= 0 or args_cli.final_window <= 0 or args_cli.hold_duration <= 0:
        raise ValueError("duration arguments must be positive")
    if args_cli.x_min >= args_cli.x_max:
        raise ValueError("--x-min must be smaller than --x-max")
    if args_cli.z_min >= args_cli.z_max:
        raise ValueError("--z-min must be smaller than --z-max")
    if args_cli.rpy_min >= args_cli.rpy_max:
        raise ValueError("--rpy-min must be smaller than --rpy-max")
    checkpoint = args_cli.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.cuda.reset_peak_memory_stats()
    all_targets = build_targets(args_cli.num_targets, target_population, args_cli.seed)
    targets = all_targets[:, args_cli.target_offset : args_cli.target_offset + args_cli.num_envs]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.seed = args_cli.seed
    env_cfg.commands.EE_pose.debug_vis = False
    env_cfg.commands.EE_pose.precision_metrics_enabled = False
    env_cfg.commands.EE_pose.resampling_time_range = (1.0e6, 1.0e6)
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if args_cli.asset_usd_dir is not None:
        env_cfg.scene.robot.spawn.usd_dir = str(args_cli.asset_usd_dir.resolve())

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    configure_agent(agent_cfg, args_cli.variant)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    runner = ImplicitOneStageRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[{args_cli.model_name}] loading checkpoint", flush=True)
    runner.load(str(checkpoint), load_optimizer=False)
    print(f"[{args_cli.model_name}] checkpoint loaded", flush=True)
    runner.eval_mode()
    command_term = env.unwrapped.command_manager.get_term("EE_pose")
    termination_manager = env.unwrapped.termination_manager
    termination_names = tuple(termination_manager.active_terms)

    step_dt = float(env.unwrapped.step_dt)
    steps_per_target = max(1, round(args_cli.target_duration / step_dt))
    final_window_steps = max(1, round(args_cli.final_window / step_dt))
    hold_steps = max(1, round(args_cli.hold_duration / step_dt))
    rows: list[dict] = []
    start_time = time.time()
    # A single explicit reset obtains the policy observations. Additional
    # top-level resets shut down this SimulationApp on Isaac Sim 4.5, so evaluate
    # a continuous, reproducible target sequence after this point.
    obs, extras = env.reset()
    print(f"[{args_cli.model_name}] environment reset complete", flush=True)
    critic_obs = extras["observations"]["critic"]
    history = extras["observations"]["contactNet"]

    for target_id in range(args_cli.num_targets):
        target_pose = install_target(command_term, targets[target_id])
        position_samples = torch.full(
            (steps_per_target, args_cli.num_envs), torch.nan, device=runner.device
        )
        orientation_samples = torch.full_like(position_samples, torch.nan)
        action_rate_samples = torch.full_like(position_samples, torch.nan)
        action_rms_samples = torch.full_like(position_samples, torch.nan)
        terminated = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=runner.device)
        termination_causes = {
            name: torch.zeros_like(terminated) for name in termination_names
        }
        hold_count = torch.zeros(args_cli.num_envs, dtype=torch.long, device=runner.device)
        succeeded = torch.zeros_like(terminated)
        previous_action = None

        for step in range(steps_per_target):
            command_term.pose_command_w.copy_(target_pose)
            command_term.time_left.fill_(1.0e6)
            with torch.inference_mode():
                action = deterministic_action(runner, obs, critic_obs, history, args_cli.variant)
                obs, _, dones, infos = env.step(action)
            critic_obs = infos["observations"]["critic"]
            history = infos["observations"]["contactNet"]
            valid = ~terminated
            position_error = command_term.metrics["position_error"]
            orientation_error = command_term.metrics["orientation_error"]
            position_samples[step] = torch.where(valid, position_error, torch.nan)
            orientation_samples[step] = torch.where(valid, orientation_error, torch.nan)
            if previous_action is not None:
                action_rate = torch.sum(torch.square(action - previous_action), dim=-1)
                action_rate_samples[step] = torch.where(valid, action_rate, torch.nan)
            action_rms = torch.sqrt(torch.mean(torch.square(action), dim=-1))
            action_rms_samples[step] = torch.where(valid, action_rms, torch.nan)
            previous_action = action.clone()
            inside = valid & (position_error <= 0.05) & (orientation_error <= math.radians(5.0))
            hold_count = torch.where(inside, hold_count + 1, torch.zeros_like(hold_count))
            succeeded |= hold_count >= hold_steps
            done_mask = dones.to(device=runner.device, dtype=torch.bool).flatten()
            for name in termination_names:
                termination_causes[name] |= termination_manager.get_term(name)
            terminated |= done_mask
            if args_cli.variant in ("transformer_gru", "gru_only"):
                runner.gru.reset_hidden_states(done_mask)

        pos_cpu = position_samples.cpu().numpy()
        ori_cpu = orientation_samples.cpu().numpy()
        action_rate_cpu = action_rate_samples.cpu().numpy()
        action_rms_cpu = action_rms_samples.cpu().numpy()
        terminated_cpu = terminated.cpu().numpy()
        termination_causes_cpu = {
            name: values.cpu().numpy() for name, values in termination_causes.items()
        }
        succeeded_cpu = succeeded.cpu().numpy()
        for env_id in range(args_cli.num_envs):
            valid_indices = np.flatnonzero(np.isfinite(pos_cpu[:, env_id]) & np.isfinite(ori_cpu[:, env_id]))
            if valid_indices.size:
                window_indices = valid_indices[-final_window_steps:]
                pos_window = pos_cpu[window_indices, env_id]
                ori_window = ori_cpu[window_indices, env_id]
                pos_mean = float(np.mean(pos_window))
                pos_p95 = float(np.quantile(pos_window, 0.95))
                ori_mean = float(np.mean(ori_window))
                ori_p95 = float(np.quantile(ori_window, 0.95))
                strict_ratio = float(
                    np.mean((pos_window <= 0.03) & (ori_window <= math.radians(5.0)))
                )
                relaxed_ratio = float(
                    np.mean((pos_window <= 0.05) & (ori_window <= math.radians(7.0)))
                )
                valid_rate = action_rate_cpu[:, env_id][np.isfinite(action_rate_cpu[:, env_id])]
                valid_rms = action_rms_cpu[:, env_id][np.isfinite(action_rms_cpu[:, env_id])]
                action_rate_mean = float(np.mean(valid_rate)) if valid_rate.size else float("nan")
                action_rms_mean = float(np.mean(valid_rms)) if valid_rms.size else float("nan")
            else:
                pos_mean = pos_p95 = ori_mean = ori_p95 = float("nan")
                strict_ratio = relaxed_ratio = 0.0
                action_rate_mean = action_rms_mean = float("nan")
            target = targets[target_id, env_id]
            reasons = [name for name in termination_names if termination_causes_cpu[name][env_id]]
            rows.append(
                {
                    "model": args_cli.model_name,
                    "variant": args_cli.variant,
                    "checkpoint": str(checkpoint),
                    "seed": args_cli.seed,
                    "env_id": env_id,
                    "target_id": target_id,
                    "target_x": float(target[0]),
                    "target_y": float(target[1]),
                    "target_z": float(target[2]),
                    "target_roll": float(target[3]),
                    "target_pitch": float(target[4]),
                    "target_yaw": float(target[5]),
                    "final_pos_mean_m": pos_mean,
                    "final_pos_p95_m": pos_p95,
                    "final_ori_mean_rad": ori_mean,
                    "final_ori_p95_rad": ori_p95,
                    "success_5cm_5deg": int(succeeded_cpu[env_id] and not terminated_cpu[env_id]),
                    "final_strict_stable_success_3cm_5deg": int(
                        strict_ratio >= 0.8 and not terminated_cpu[env_id]
                    ),
                    "final_relaxed_stable_success_5cm_7deg": int(
                        relaxed_ratio >= 0.8 and not terminated_cpu[env_id]
                    ),
                    "final_strict_in_threshold_ratio": strict_ratio,
                    "final_relaxed_in_threshold_ratio": relaxed_ratio,
                    "terminated": int(terminated_cpu[env_id]),
                    "termination_reason": ";".join(reasons),
                    "action_rate_l2_mean": action_rate_mean,
                    "action_rms": action_rms_mean,
                }
            )
        print(f"[{args_cli.model_name}] target batch {target_id + 1}/{args_cli.num_targets}", flush=True)

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with args_cli.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows, targets)
    summary["wall_time_s"] = time.time() - start_time
    summary_path = args_cli.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
