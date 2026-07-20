from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
# from isaaclab.envs.mdp.events import _randomize_prop_by_op  # noqa: F401, F403
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from ext_loco.utils.math import generate_sigmoid_scale


def reset_selected_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset all joints to default and randomize only the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_ids = asset_cfg.joint_ids

    selected_pos = joint_pos[:, joint_ids]
    selected_vel = joint_vel[:, joint_ids]
    joint_pos[:, joint_ids] = selected_pos + sample_uniform(
        *position_range, selected_pos.shape, selected_pos.device
    )
    joint_vel[:, joint_ids] = selected_vel + sample_uniform(
        *velocity_range, selected_vel.shape, selected_vel.device
    )

    joint_pos.clamp_(
        asset.data.soft_joint_pos_limits[env_ids, :, 0],
        asset.data.soft_joint_pos_limits[env_ids, :, 1],
    )
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def upate_EE_reference_and_cb(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    decrese_velocity: tuple[float, float],  # TODO, make this value random for different envs.
    cb_incease_velocity: float = 0.5,
    command_name: str = "EE_pose",
):
    """Reset the reference SE(3) distance of the end-effector."""

    # check whether asset is a copy or a view of env.scene[asset_cfg.name]

    if not hasattr(env, "EE_se3_cb_error"):
        env.EE_se3_cb_error = torch.zeros(env.scene.num_envs, device=env.device)

    if not hasattr(env, "EE_se3_distance_reference"):
        env.EE_se3_distance_reference = torch.ones(env.scene.num_envs, 1, device=env.device) * 10

    if not hasattr(env, "last_EE_se3_commands_w"):
        env.last_EE_se3_commands_w = torch.zeros(env.scene.num_envs, 7, device=env.device)

    # update penalty
    sigmoid_scale = generate_sigmoid_scale(mu=0.4, decay_length=0.4, x=env.EE_se3_distance_reference.squeeze(1))

    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    EE_orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    EE_pose_command_w = env.command_manager.get_term(command_name).pose_command_w # type: ignore

    env.EE_se3_cb_error += (
        (1 - sigmoid_scale) * cb_incease_velocity * env.step_dt * (2 * EE_position_error + EE_orientation_error)
    )

    # release penalty
    release_pos = (0.2 - 2 * EE_position_error).clip(min=0.0, max=0.2)

    release_orient = (0.2 - EE_orientation_error).clip(min=0.0, max=0.2)

    env.EE_se3_cb_error -= (1 - sigmoid_scale) * cb_incease_velocity * env.step_dt * (release_pos + release_orient)
    env.EE_se3_cb_error = torch.clip(env.EE_se3_cb_error, min=0.0, max=1.2)

    env.EE_se3_distance_reference -= env._se3_decrease_vel * env.step_dt
    env.EE_se3_distance_reference = torch.clip(env.EE_se3_distance_reference, min=0.0)

    reset_mask = torch.any((EE_pose_command_w - env.last_EE_se3_commands_w).abs() > 1e-1, dim=-1)

    env.last_EE_se3_commands_w = EE_pose_command_w.clone()

    env._se3_decrease_vel = math_utils.sample_uniform(
        decrese_velocity[0], decrese_velocity[1], env.scene.num_envs, env.device
    )

    env.EE_se3_distance_reference[reset_mask, 0] = 2 * EE_position_error[reset_mask] + EE_orientation_error[reset_mask]
    env.EE_se3_cb_error[reset_mask] = 0.0


def prepare_quantity_for_tron1_piper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Compute the nominal foot position in the body frame.

    This function computes the nominal foot position in the body frame. This funciton is only suitable for LIMX W1 robot

    The computed nominal foot position is stored in the attribute ``env._nominal_foot_position_b`` of the asset.
    """
    # extract the used quantities (to enable type-hinting)
    
    asset: Articulation = env.scene[asset_cfg.name]

    try:
        wheel_link_idx, _ = asset.find_bodies("wheel_[RL]_Link")
        wheel_joint_ids, _ = asset.find_joints("wheel_[RL]_Joint")
    except ValueError:
        wheel_link_idx, _ = asset.find_bodies("ankle_[RL]_Link")
        wheel_joint_ids, _ = asset.find_joints("ankle_[RL]_Joint")

    arm_joint_idx, _ = asset.find_joints(r"J\d")

    ee_link_idx, _ = asset.find_bodies("eef_link")

    base_idx, _ = asset.find_bodies("base_Link")

    # hip_idx, _ = asset.find_bodies(".*_thigh")

    wheels_pos_w = asset.data.body_pos_w[:, wheel_link_idx, :]

    base_pos_w = asset.data.body_pos_w[:, base_idx, :]

    # hip_pos_w = asset.data.body_pos_w[:, hip_idx, :]

    base_quat = asset.data.body_quat_w[:, base_idx, :]

    nominal_foot_position_b = torch.zeros(len(wheel_link_idx), 3, device=asset.device)
    # wheels_pos_b = torch.zeros(asset.num_instances, len(wheel_idx), 3, device=asset.device)

    for j in range(asset.num_instances):
        if torch.any(asset.data.joint_pos[j, :] > 5e-2):
            continue
        for i in range(len(wheel_link_idx)):
            nominal_foot_position_b[i, :] = math_utils.quat_rotate_inverse(
                base_quat[j, 0, :], wheels_pos_w[j, i, :] - base_pos_w[j, 0, :]
            )
            # nominal_foot_position_b[i, 1] = math_utils.quat_rotate_inverse(
            #     base_quat[j, 0, :], wheels_pos_w[j, i, :] - base_pos_w[j, 0, :]
            # )[1]
        break
    # assert (nominal_foot_position_b != 0.0).any()
    
    env._nominal_foot_position_b = nominal_foot_position_b # type: ignore
    env._wheels_link_ids = wheel_link_idx                  # type: ignore
    env._wheels_joint_ids = wheel_joint_ids                # type: ignore
    env._arm_joint_ids = arm_joint_idx                     # type: ignore
    env._ee_link_idx = ee_link_idx                         # type: ignore
    env._foot_radius = 0.127                               # type: ignore
