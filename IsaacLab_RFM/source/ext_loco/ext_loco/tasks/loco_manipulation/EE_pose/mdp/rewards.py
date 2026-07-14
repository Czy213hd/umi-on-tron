from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any, Dict

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ext_loco.utils.math import generate_sigmoid_scale, relexed_barrier_func, command_duration_mask


def tracking_EE_cb_penalty_l1(
    env: ManagerBasedRLEnv,
):
    """Penalty cumulative tracking error of EE SE3 commands using L1 kernel."""
    # compute the error
    return env.EE_se3_cb_error


def safety_reward_exp(
    env: ManagerBasedRLEnv,
    std: float,
    base_height_target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward safety of EE position commands using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # prepare variable
    # cache foot link ids (set by observations.foot_position_b)
    if not hasattr(env, "_foot_link_ids"):
        raise AttributeError("Foot link ids not set on env. Ensure observations.foot_position_b is called first.")
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_position = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)

    # check if root_state equal
    # if asset.data.root_quat_w != base_quat:
    #     raise ValueError("Root state is not equal to base state")

    """Compute the final error"""
    # compute the nominal foot error
    foot_position = asset.data.body_pos_w[:, env._foot_link_ids, :]
    foot_position_b = math_utils.quat_rotate_inverse(base_quat, foot_position - base_position)
    # base_height = -foot_position_b[:, :, 2].mean(dim=-1) + env._foot_radius
    base_height = asset.data.root_link_pos_w[:, 2]

    foot_pos_error_b = foot_position_b[:, :, :2] - env._nominal_foot_position_b[:, :2]
    # inner eight condition
    inner_eight_condition = ((env._nominal_foot_position_b[:, 1] > 0.0) * (foot_pos_error_b[:, :, 1] < 0.0)) | (
        (env._nominal_foot_position_b[:, 1] < 0.0) * (foot_pos_error_b[:, :, 1] > 0.0)
    )

    foot_pos_error_b[:, :, 1] = torch.where(
        inner_eight_condition, foot_pos_error_b[:, :, 1] / 0.1, foot_pos_error_b[:, :, 1] / 0.2
    )
    foot_pos_error_b[:, :, 0] = foot_pos_error_b[:, :, 0] / 0.2

    foot_pos_error_b = torch.sum(torch.sum(foot_pos_error_b.abs(), dim=-1), dim=-1)
    foot_pos_error_b = torch.clamp(foot_pos_error_b, max=8.0)
    # compute base error
    base_orient_error_roll = torch.abs(asset.data.projected_gravity_b[:, 1]) / 0.1
    base_orient_error_pitch = torch.abs(asset.data.projected_gravity_b[:, 0]) / 0.85
    base_height_error = torch.abs((base_height - base_height_target)) / 0.2  # not used
    # arm_middle_pos = torch.tensor([0.0, 1.57, -2.7, 0.0, 0.0, 0.0], device=asset.device)
    arm_middle_pos = (
        asset.data.default_joint_limits[:, env._arm_joint_ids, 1]
        + asset.data.default_joint_limits[:, env._arm_joint_ids, 0]
    ) / 2.0
    # arm in middle position
    arm_middle_error = (
        torch.sum(
            torch.abs(
                torch.index_select(
                    asset.data.joint_pos,
                    1,
                    torch.tensor(env._arm_joint_ids, device=asset.device),
                )
                - arm_middle_pos
            ),
            dim=1,
        )
        / 9.0
    )

    # body velocity penalty
    wheel_vel_error = (torch.sum(torch.abs(asset.data.joint_vel[:, env._wheels_joint_ids]), dim=1) / 3.0).clip(max=4)
    base_lin_vel_error = torch.norm(asset.data.root_link_lin_vel_b, p=2, dim=1) / 0.5
    base_ang_vel_error = torch.norm(asset.data.root_link_ang_vel_b, p=2, dim=1) / 1.2
    Ee_lin_vel_error = torch.norm(asset.data.body_lin_vel_w[:, env._ee_link_idx].squeeze(1), p=2, dim=1) / 0.6
    Ee_ang_vel_error = torch.norm(asset.data.body_ang_vel_w[:, env._ee_link_idx].squeeze(1), p=2, dim=1) / 3.14

    normalized_mani_error = (
        foot_pos_error_b  # 2
        + wheel_vel_error  # 2
        + base_lin_vel_error  # 1
        + base_ang_vel_error  # 1
        + base_height_error * 0.5  # 1
        + base_orient_error_roll * 0.5  # 0.5
        + base_orient_error_pitch * 0.25  # 0.5
        + arm_middle_error  # 1
        + Ee_lin_vel_error  # 1
        + Ee_ang_vel_error  # 1
    ) / 13.0

    normalized_loco_error = (
        foot_pos_error_b / 2.0  # 2
        + base_orient_error_pitch  # 0.5
        + base_orient_error_roll  # 0.5
        # + arm_middle_error * 1.5  # 1
        + base_height_error  # 1
    ) / 6.0

    # for debug
    # total = foot_pos_error.mean() + base_orient_error.mean() + arm_middle_error.mean() + base_height_error.mean()
    # foot_error_ratio = foot_pos_error.mean() / total
    # base_error_ratio = base_orient_error.mean() / total
    # arm_error_ratio = arm_middle_error.mean() / total
    # height_error_ratio = base_height_error.mean() / total
    # print(
    #     f"foot_error_ratio: {foot_error_ratio}, \n base_error_ratio: {base_error_ratio}, \n arm_error_ratio: {arm_error_ratio}, \n height_error_ratio: {height_error_ratio}"
    # )
    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )
    mani_safety_scale = torch.exp(-normalized_mani_error / std**2)

    loco_safety_scale = torch.exp(-normalized_loco_error / std**2)

    env._mani_safety_scale = mani_safety_scale + 0.4

    env._loco_safety_scale = loco_safety_scale + 0.4

    return mani_safety_scale * (1 - env._loco_mani_scale) + loco_safety_scale * env._loco_mani_scale


def track_EE_reference_exp(
    env: ManagerBasedRLEnv,
    std: float,
    relese_delta: float = 0.5,
    init_value: float = 0.99,
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """Reward tracking of EE SE3 reference using exponential kernel."""
    # compute the error

    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )
    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)

    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    EE_orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    se3_distance_ref = env.command_manager.get_term("EE_pose").se3_distance_ref

    track_error = torch.abs(se3_distance_ref - EE_orientation_error - 2 * EE_position_error) - relese_delta

    track_error = torch.clamp(track_error, min=0.0)

    return torch.exp(-track_error / std**2) * env._loco_mani_scale * env._loco_safety_scale


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    is_contact_num = torch.sum(torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold, dim=-1)
    # if getattr(env, "_feet_ids", None) is None:
    #     env._feet_ids = contact_sensor.find_bodies("wheel_.*")[0]
    # feet_is_contact = torch.sum(
    #     torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, env._feet_ids, :2], dim=-1), dim=1)[0]
    #     > threshold,
    #     dim=-1,
    # )
    # sum over contacts for each environment
    return is_contact_num  # + feet_is_contact


def weighted_joint_torques_l2(
    env: ManagerBasedRLEnv,
    torque_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    weighted_torque = torch.zeros_like(asset.data.applied_torque)

    for joint_name, w in torque_weight.items():
        joint_idx, _ = asset.find_joints(joint_name)
        weighted_torque[:, joint_idx] = torch.square(asset.data.applied_torque[:, joint_idx]) * w

    return torch.sum(weighted_torque, dim=1)


def weighted_joint_power_l1(
    env: ManagerBasedRLEnv,
    power_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint power applied on the articulation using L1 kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    weighted_power = torch.zeros_like(asset.data.applied_torque)

    for joint_name, w in power_weight.items():
        joint_idx = asset.find_joints(joint_name)[0]
        weighted_power[:, joint_idx] = (
            torch.abs(asset.data.applied_torque[:, joint_idx] * asset.data.joint_vel[:, joint_idx]) * w
        )
    # if hasattr(env, "_wheels_joint_ids"):
    #     if torch.rand(1) < 0.1:
    #         print(
    #             (
    #                 torch.abs(
    #                     asset.data.applied_torque[:, env._wheels_joint_ids]
    #                     * asset.data.joint_vel[:, env._wheels_joint_ids]
    #                     * 2
    #                 ).sum(dim=-1)
    #                 / torch.sum(weighted_power, dim=1)
    #             ).mean()
    #         )

    return torch.sum(weighted_power, dim=1)


def track_EE_position_exp(
    env: ManagerBasedRLEnv,
    std: float,
    init_value: float = 0.99,
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """Reward tracking of EE position commands using exponential kernel."""
    # compute the error
    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)

    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]

    normal = torch.exp(-EE_position_error / std**2)

    micro_enhancement = torch.exp(-5 * EE_position_error / std**2)

    return (normal + micro_enhancement) * (1 - env._loco_mani_scale) * env._mani_safety_scale


def track_EE_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "EE_pose",
    init_value: float = 0.99,
) -> torch.Tensor:
    """Reward tracking of EE orientation commands using exponential kernel."""

    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)

    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # over_contact_mask = (
    #     torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    #     > threshold
    # ).any(dim=-1)

    # position_tracking_priority
    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]

    position_scale = torch.exp(-EE_position_error / 0.5)

    EE_orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]

    normal = torch.exp(-EE_orientation_error / std**2)

    micro_enhancement = torch.exp(-5 * EE_orientation_error / std**2)

    return (normal + micro_enhancement) * position_scale * (1 - env._loco_mani_scale)  * env._mani_safety_scale


def reach_EE_position(
    env: ManagerBasedRLEnv, std: float, duration: float = 3, command_name: str = "EE_pose", init_value: float = 0.99
) -> torch.Tensor:
    """Reward tracking of EE position commands using exponential kernel."""
    # compute the error
    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )
    EE_position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    time_left = env.command_manager.get_term(command_name).time_left
    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)
    mask = command_duration_mask(time_left, duration)
    normal = (1.0 / (1.0 + torch.square(EE_position_error / std**2))) * mask
    micro_enhancement = (1.0 / (1.0 + torch.square(4 * EE_position_error / std**2))) * mask
    return (normal + micro_enhancement) * env._mani_safety_scale * (1 - env._loco_mani_scale)


def reach_EE_orient(
    env: ManagerBasedRLEnv, std: float, duration: float = 3, command_name: str = "EE_pose", init_value: float = 0.99
) -> torch.Tensor:
    """Reward tracking of EE position commands using exponential kernel."""
    # compute the error
    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )
    # contact_time = env.command_manager.get_term(command_name).metrics["contact_time"]
    # init_value = torch.ones(env.num_envs, device=env.device) * init_value
    # scale = torch.pow(init_value, contact_time)
    EE_orient_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    time_left = env.command_manager.get_term(command_name).time_left
    mask = command_duration_mask(time_left, duration)
    normal = (1.0 / (1.0 + torch.square(EE_orient_error / std**2))) * mask
    micro_enhancement = (1.0 / (1.0 + torch.square(4 * EE_orient_error / std**2))) * mask
    return (normal + micro_enhancement) * env._mani_safety_scale * (1 - env._loco_mani_scale)


def track_EE_pb(env: ManagerBasedRLEnv, command_name: str = "EE_pose") -> torch.Tensor:
    """Reward tracking of EE position commands using exponential kernel."""
    # compute the error
    optim_pos_distance = env.command_manager.get_term(command_name).optim_pos_distance
    position_scale = torch.exp(-optim_pos_distance / 0.5)
    optim_orient_distance = env.command_manager.get_term(command_name).optim_orient_distance
    orient_scale = torch.exp(-optim_orient_distance / 0.5)
    pos_improve = env.command_manager.get_term(command_name).pos_improvement
    orient_improve = env.command_manager.get_term(command_name).orient_improvement
    return (2 * pos_improve * position_scale + orient_improve * orient_scale) * env._loco_safety_scale


def body_ee_alignment(env: ManagerBasedRLEnv, joint_names: list[str] = ("J1", "J4")) -> torch.Tensor:
    """Penalty to keep yaw-dominant arm joints close to their default (align EE with body)."""
    asset: Articulation = env.scene["robot"]
    joint_ids = asset.find_joints(joint_names)[0]
    diff = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(diff.abs(), dim=1)


def base_bidirectional_target_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "EE_pose",
    min_target_distance: float = 0.1,
) -> torch.Tensor:
    """Align the base longitudinal axis with the horizontal EE-target direction.

    At command sampling, a target in the front half-plane is assigned +1 and
    one in the rear half-plane -1. The fixed mode produces the penalty
    ``1 - direction * cos(heading_error)``. Rear targets therefore favor
    backward walking and penalize a later 180-degree turn.
    """
    asset: Articulation = env.scene["robot"]
    command = env.command_manager.get_term(command_name)

    target_delta_xy = command.pose_command_w[:, :2] - asset.data.root_link_pos_w[:, :2]
    target_distance = torch.linalg.vector_norm(target_delta_xy, dim=1)
    target_direction = target_delta_xy / target_distance.clamp_min(1.0e-6).unsqueeze(1)

    local_forward = torch.zeros(env.num_envs, 3, device=env.device)
    local_forward[:, 0] = 1.0
    base_forward_w = math_utils.quat_apply(asset.data.root_link_quat_w, local_forward)
    base_forward_xy = base_forward_w[:, :2]
    base_forward_xy = base_forward_xy / torch.linalg.vector_norm(
        base_forward_xy, dim=1
    ).clamp_min(1.0e-6).unsqueeze(1)

    heading_cosine = torch.sum(base_forward_xy * target_direction, dim=1).clamp(-1.0, 1.0)
    penalty = 1.0 - command.travel_direction * heading_cosine
    valid_target = (target_distance >= min_target_distance).float()
    return penalty * valid_target * env._loco_mani_scale


def legs_min_separation(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.3,
    body_names: tuple[str, str] = ("ankle_L_Link", "ankle_R_Link"),
    axis: str = "y",
) -> torch.Tensor:
    """Penalty if the two legs (ankles/feet) get closer than a target distance along one base-frame axis."""
    asset: Articulation = env.scene["robot"]
    # cache leg link ids
    if not hasattr(env, "_leg_link_ids"):
        env._leg_link_ids = asset.find_bodies(list(body_names))[0]  # type: ignore
    ids = env._leg_link_ids
    axis_id = {"x": 0, "y": 1}[axis]
    foot_pos_w = asset.data.body_pos_w[:, ids, :]  # (N,2,3)
    base_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos_w = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)
    foot_pos_b = math_utils.quat_rotate_inverse(base_quat_w, foot_pos_w - base_pos_w)
    dist = torch.abs(foot_pos_b[:, 0, axis_id] - foot_pos_b[:, 1, axis_id])
    penalty = torch.clamp(min_distance - dist, min=0.0)
    return penalty


def foot_flat_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize ankle/foot links whose local z-axis tilts away from the world vertical axis."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    local_z = torch.zeros(foot_quat_w.shape[0], foot_quat_w.shape[1], 3, device=asset.device)
    local_z[..., 2] = 1.0

    foot_z_w = math_utils.quat_apply(foot_quat_w.reshape(-1, 4), local_z.reshape(-1, 3)).reshape_as(local_z)
    # Equivalent to penalizing roll/pitch of the foot link while allowing yaw.
    return torch.sum(torch.square(foot_z_w[..., :2]), dim=1).sum(dim=1)


def foot_slip_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal foot velocity while the foot is in contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    is_contact = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1)[0] > threshold
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * is_contact, dim=1)


def pose_product_reward(
    env: ManagerBasedRLEnv,
    pos_sigma: float,
    orn_sigma: float,
    command_name: str = "EE_pose",
) -> torch.Tensor:
    """UMI-style EE pose reward: position and orientation terms multiplied."""
    term = env.command_manager.get_term(command_name)
    pos_err = term.metrics["position_error"]
    orn_err = term.metrics["orientation_error"]
    pos_reward = torch.exp(-(pos_err ** 2) / pos_sigma)
    orn_reward = torch.exp(-orn_err / orn_sigma)
    return pos_reward * orn_reward


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = asset.data.root_link_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # sensor.data.ray_hits_w can be inf, so we clip it to avoid NaN
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    return torch.square(height.mean(dim=1) - target_height)


def flat_orientation_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 2] + 1)


def dof_power_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power applied on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )


def dof_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint power applied on the articulation using L2-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the L2 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]),
        dim=1,
    )


def feet_regulation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    feet_radius: float,
    feet_height_target: float,
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - feet_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    reward = torch.sum(
        torch.exp(-feet_height / feet_height_target) * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1
    )
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_contacts_reg(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Regulate feet contacts"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    # matrix_contact_forces = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, 0, :]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # feet_contact_num = torch.sum(torch.norm(matrix_contact_forces, dim=-1) > threshold, dim=-1)
    # reward_loco = (feet_contact_num == 2) | (feet_contact_num == 4)
    reward_mani = torch.sum(is_contact, dim=-1) == 2

    EE_position_error = env.command_manager.get_term("EE_pose").metrics["position_error"]

    position_scale = torch.exp(-EE_position_error / 0.5)

    # sigmoid_scale = generate_sigmoid_scale(
    #     mu=0.8, decay_length=0.8, x=env.command_manager.get_term("EE_pose").se3_distance_ref
    # )

    return reward_mani * (1 - env._loco_mani_scale) * position_scale


def fly_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Regulate feet contacts"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w

    feet_contact_num = torch.sum(torch.norm(net_contact_forces, dim=-1) > threshold, dim=-1)
    reward = feet_contact_num == 0

    return reward


def no_fly(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    return 1 - fly_penalty(env, threshold, sensor_cfg).float()


class ActionSmoothnessPenaltyWrapper:
    """
    A wrapper class for calculating action smoothness penalty.

    The main purposes of this wrapper are:
    1. To maintain state across multiple calls (prev_action and prev_prev_action).
    2. To calculate a smoothness penalty based on the current, previous, and
       two-steps-ago actions.
    3. To provide a serializable interface compatible with IsaacLab's YAML
       configuration system.
    """

    def __init__(self):
        self.prev_prev_action = None
        self.prev_action = None
        self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalize large instantaneous changes in the network action output"""
        current_action = env.action_manager.action.clone()

        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update actions for next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        startup_env_musk = env.episode_length_buf < 3
        penalty[startup_env_musk] = 0

        return penalty


action_smoothness_penalty = ActionSmoothnessPenaltyWrapper()
