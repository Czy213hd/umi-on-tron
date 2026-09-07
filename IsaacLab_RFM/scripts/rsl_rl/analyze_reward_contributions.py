#!/usr/bin/env python3
"""Measure gated/ungated reward contributions on one shared baseline-policy rollout."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--duration", type=float, default=8.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", default="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0")
parser.add_argument("--asset-usd-dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import json
import math

import gymnasium as gym
import torch

import ext_loco.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import ImplicitOneStageRunner


UNGATED_PARAMS = {
    "safety_exp": {"use_gates": False},
    "track_EE_position_exp": {"use_gates": False},
    "track_EE_orientation_fine_exp": {"use_gates": False},
    "track_EE_pb": {"use_walking_gate": False, "use_safety_gate": False},
    "feet_contacts_reg": {"use_standing_gate": False, "use_position_gate": False},
    "foot_flat_l2": {"use_standing_gate": False},
}


def deterministic_action(runner, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    normalized_obs = runner.obs_normalizer(obs)
    normalized_history = runner.contactNet_obs_normalizer(history)
    encoder_output = runner.contactNet(normalized_history)
    encoded = runner.gru.gru_forward(encoder_output, runner.gru.hidden_state)
    latent_dim = runner.ppo_alg.next_obs_latent_dim
    actor_input = torch.cat(
        (normalized_obs, encoded[:, :3], encoded[:, 3 : 3 + latent_dim]), dim=-1
    )
    return runner.actor_critic.act_inference(actor_input)


def make_accumulator(term_names: list[str]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "gated_raw_sum": 0.0,
            "gated_weighted_sum": 0.0,
            "gated_abs_weighted_sum": 0.0,
            "ungated_raw_sum": 0.0,
            "ungated_weighted_sum": 0.0,
            "ungated_abs_weighted_sum": 0.0,
        }
        for name in term_names
    }


def main() -> None:
    checkpoint = args_cli.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args_cli.num_envs <= 0 or args_cli.duration <= 0:
        raise ValueError("--num-envs and --duration must be positive")

    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env_cfg.seed = args_cli.seed
    env_cfg.commands.EE_pose.debug_vis = False
    env_cfg.commands.EE_pose.precision_metrics_enabled = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if args_cli.asset_usd_dir is not None:
        env_cfg.scene.robot.spawn.usd_dir = str(args_cli.asset_usd_dir.resolve())

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    runner = ImplicitOneStageRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    runner.eval_mode()

    reward_manager = env.unwrapped.reward_manager
    term_names = list(reward_manager.active_terms)
    term_cfgs = {name: reward_manager.get_term_cfg(name) for name in term_names}
    accum = make_accumulator(term_names)
    sample_count = 0
    obs, extras = env.reset()
    history = extras["observations"]["contactNet"]
    steps = max(1, round(args_cli.duration / float(env.unwrapped.step_dt)))

    for _ in range(steps):
        with torch.inference_mode():
            action = deterministic_action(runner, obs, history)
            obs, _, dones, infos = env.step(action)
        history = infos["observations"]["contactNet"]
        sample_count += args_cli.num_envs

        for idx, name in enumerate(term_names):
            cfg = term_cfgs[name]
            weight = float(cfg.weight)
            gated_weighted = reward_manager._step_reward[:, idx]
            gated_raw = gated_weighted / weight if weight != 0 else torch.zeros_like(gated_weighted)

            if name in UNGATED_PARAMS:
                params = deepcopy(cfg.params)
                params.update(UNGATED_PARAMS[name])
                with torch.inference_mode():
                    ungated_raw = cfg.func(env.unwrapped, **params)
            else:
                ungated_raw = gated_raw
            ungated_weighted = ungated_raw * weight

            values = accum[name]
            values["gated_raw_sum"] += gated_raw.sum().item()
            values["gated_weighted_sum"] += gated_weighted.sum().item()
            values["gated_abs_weighted_sum"] += gated_weighted.abs().sum().item()
            values["ungated_raw_sum"] += ungated_raw.sum().item()
            values["ungated_weighted_sum"] += ungated_weighted.sum().item()
            values["ungated_abs_weighted_sum"] += ungated_weighted.abs().sum().item()

        done_mask = dones.to(device=runner.device, dtype=torch.bool).flatten()
        runner.gru.reset_hidden_states(done_mask)

    rows = []
    gated_abs_total = sum(v["gated_abs_weighted_sum"] for v in accum.values()) / sample_count
    ungated_abs_total = sum(v["ungated_abs_weighted_sum"] for v in accum.values()) / sample_count
    gated_signed_total = sum(v["gated_weighted_sum"] for v in accum.values()) / sample_count
    ungated_signed_total = sum(v["ungated_weighted_sum"] for v in accum.values()) / sample_count

    for name in term_names:
        cfg = term_cfgs[name]
        values = accum[name]
        gated_abs = values["gated_abs_weighted_sum"] / sample_count
        ungated_abs = values["ungated_abs_weighted_sum"] / sample_count
        ratio = gated_abs / ungated_abs if ungated_abs > 1.0e-12 else 1.0
        matched_weight = float(cfg.weight) * ratio if name in UNGATED_PARAMS else float(cfg.weight)
        rows.append(
            {
                "term": name,
                "is_gate_ablated": name in UNGATED_PARAMS,
                "original_weight": float(cfg.weight),
                "gated_raw_mean": values["gated_raw_sum"] / sample_count,
                "gated_weighted_mean": values["gated_weighted_sum"] / sample_count,
                "gated_abs_weighted_mean": gated_abs,
                "gated_signed_share": (values["gated_weighted_sum"] / sample_count) / gated_signed_total
                if abs(gated_signed_total) > 1.0e-12 else math.nan,
                "gated_abs_share": gated_abs / gated_abs_total if gated_abs_total > 0 else math.nan,
                "ungated_raw_mean": values["ungated_raw_sum"] / sample_count,
                "ungated_weighted_mean": values["ungated_weighted_sum"] / sample_count,
                "ungated_abs_weighted_mean": ungated_abs,
                "ungated_signed_share": (values["ungated_weighted_sum"] / sample_count) / ungated_signed_total
                if abs(ungated_signed_total) > 1.0e-12 else math.nan,
                "ungated_abs_share": ungated_abs / ungated_abs_total if ungated_abs_total > 0 else math.nan,
                "match_ratio": ratio,
                "matched_weight": matched_weight,
            }
        )

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args_cli.output_dir / "reward_contributions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    overrides = []
    for name, params in UNGATED_PARAMS.items():
        for key, value in params.items():
            overrides.append(f"env.rewards.{name}.params.{key}={str(value).lower()}")
        matched = next(row["matched_weight"] for row in rows if row["term"] == name)
        overrides.append(f"env.rewards.{name}.weight={matched:.10g}")
    (args_cli.output_dir / "matched_additive_overrides.txt").write_text(
        "\n".join(overrides) + "\n", encoding="utf-8"
    )
    summary = {
        "checkpoint": str(checkpoint),
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": steps,
        "state_samples": sample_count,
        "gated_signed_total_mean": gated_signed_total,
        "gated_abs_total_mean": gated_abs_total,
        "ungated_signed_total_mean": ungated_signed_total,
        "ungated_abs_total_mean": ungated_abs_total,
    }
    (args_cli.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {csv_path}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
