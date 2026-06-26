from __future__ import annotations
from ast import List

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, Imu

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint torques of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def joint_measured_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint torques of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    asset.root_physx_view.get_dof_projected_joint_forces()
    return asset.root_physx_view.get_dof_projected_joint_forces()[:, asset_cfg.joint_ids]


def joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint acceleration of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def time_left(env: ManagerBasedRLEnv):
    return env.command_manager.get_term("EE_pose").time_left.unsqueeze(-1)


def foot_position_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wheel_.*")
) -> torch.Tensor:
    """The foot position of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # cache foot link ids
    if not hasattr(env, "_foot_link_ids"):
        if asset_cfg.body_names is None:
            raise ValueError("body_names cannot be None")
        env._foot_link_ids = asset.find_bodies(asset_cfg.body_names)[0]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_position = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)
    foot_position = asset.data.body_pos_w[:, env._foot_link_ids, :]
    foot_position_b = math_utils.quat_rotate_inverse(base_quat, foot_position - base_position)
    return foot_position_b.flatten(1, 2)


def body_any_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), vel_type: str = "linear"
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 1:
        raise ValueError(f"Only one body id is allowed, got {len(asset_cfg.body_ids)}")
    if vel_type == "linear":
        body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    elif vel_type == "angular":
        body_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    else:
        raise ValueError(f"Unknown velocity type: {vel_type}")
    body_vel_b = math_utils.quat_rotate_inverse(asset.data.root_link_quat_w, body_vel.squeeze(1))
    return body_vel_b


def EE_current_pose_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end-effector")
) -> torch.Tensor:
    """The current pose of the end-effector of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    EE_position_b = math_utils.quat_rotate_inverse(
        asset.data.root_link_quat_w,
        (asset.data.body_link_state_w[:, asset_cfg.body_ids, :3].squeeze(1) - asset.data.root_link_pos_w),
    )
    q_se = math_utils.quat_unique(asset.data.body_link_state_w[:, asset_cfg.body_ids, 3:7].squeeze(1)) # ee 在 世界坐标系下的四元数
    q_sb = math_utils.quat_unique(asset.data.root_link_quat_w) # 机器人基座在 世界坐标系下的四元数
    q_be = math_utils.quat_mul(math_utils.quat_inv(q_sb), q_se) # ee 在 机器人基座坐标系下的四元数
    EE_orientation_b = math_utils.matrix_from_quat(math_utils.quat_unique(q_be))

    EE_orient_x = EE_orientation_b[:, :, 0] # ee 在 机器人基座坐标系下的 x 轴方向
    EE_orient_y = EE_orientation_b[:, :, 1] # ee 在 机器人基座坐标系下的 y 轴方向
    # print('EE_position_b: ', EE_position_b)
    # print('EE_orient_x: ', EE_orient_x)
    # print('EE_orient_y: ', EE_orient_y)
    return torch.cat((EE_position_b, EE_orient_x, EE_orient_y), dim=-1)


def EE_commands_b(
    env: ManagerBasedRLEnv,
):
    EE_pose_b = env.command_manager.get_command("EE_pose")

    EE_orientation = math_utils.matrix_from_quat(math_utils.quat_unique(EE_pose_b[:, 3:7]))
    EE_orientation_x = EE_orientation[:, :, 0]
    EE_orientation_y = EE_orientation[:, :, 1]
    return torch.cat((EE_pose_b[:, :3], EE_orientation_x, EE_orientation_y), dim=-1)


def EE_se3_distance_ref(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The SE(3) distance of the end-effector"""
    return env.command_manager.get_term("EE_pose").se3_distance_ref.unsqueeze(-1) # type: ignore


def EE_se3_cb_error(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The SE(3) tracking cb error of the end-effector"""

    if not hasattr(env, "EE_se3_cb_error"):
        env.EE_se3_cb_error = torch.zeros(env.scene.num_envs, device=env.device)

    return env.EE_se3_cb_error.unsqueeze(1)


def joint_pos_rel_exclude_wheel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="(?!.*_WHL).+")
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def base_pos_z_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.55
) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2].unsqueeze(-1) - target_height


def default_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The default joint positions of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def default_joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The default joint velocities of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_joint_vel[:, asset_cfg.joint_ids]


def joint_kp(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The default joint velocities of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness[:, asset_cfg.joint_ids]


def joint_kd(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The default joint velocities of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping[:, asset_cfg.joint_ids]


class JointPosDiffWrapper:
    def __init__(self):
        self.joint_pos_last = None
        self.__name__ = "joint_pos_diff"

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The mass of the body of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]

        if self.joint_pos_last is None:
            self.joint_pos_diff = asset.data.joint_vel[:, asset_cfg.joint_ids]
            self.joint_pos_last = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()
            return self.joint_pos_diff

        self.joint_pos_diff = (asset.data.joint_pos[:, asset_cfg.joint_ids] - self.joint_pos_last) / env.step_dt
        self.joint_pos_last = asset.data.joint_pos[:, asset_cfg.joint_ids].clone()

        startup_env_musk = env.episode_length_buf == 0
        self.joint_pos_diff[startup_env_musk] = asset.data.joint_vel[startup_env_musk, asset_cfg.joint_ids]

        return self.joint_pos_diff


joint_pos_diff = JointPosDiffWrapper()


class BodyMassWrapper:
    def __init__(self):
        self.count = 0
        self.masses = None
        self.__name__ = "body_mass"

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The mass of the body of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.count <= 1:
            self.masses = asset.root_physx_view.get_masses().to(device=asset.device)

        self.count += 1
        # Return the mass of the base for all environments
        return self.masses[:, asset_cfg.body_ids]


body_mass = BodyMassWrapper()


class BodyMassRelWrapper:
    def __init__(self):
        self.count = 0
        self.masses = None
        self.default_masses = None
        self.__name__ = "body_mass_rel"

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The mass of the body of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.count <= 1:
            self.masses = asset.root_physx_view.get_masses().to(device=asset.device)
            self.default_masses = asset.data.default_mass.to(device=asset.device)

        self.count += 1
        # Return the mass of the base for all environments
        return self.masses[:, asset_cfg.body_ids] - self.default_masses[:, asset_cfg.body_ids]


body_mass_rel = BodyMassRelWrapper()


class RigidBodyMaterialsWrapper:
    def __init__(
        self,
    ):
        self.count = 0
        self.materials = None
        self.num_shapes_per_body = []
        self.__name__ = "rigid_body_materials"
        self.asset = None

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The mass of the body of the asset."""
        # extract the used quantities (to enable type-hinting)
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.asset is None:
            self.asset = asset
            if isinstance(self.asset, Articulation):
                self.num_shapes_per_body = []
                for link_path in self.asset.root_physx_view.link_paths[0]:
                    link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                    self.num_shapes_per_body.append(link_physx_view.max_shapes)
                # ensure the parsing is correct
                num_shapes = sum(self.num_shapes_per_body)
                expected_shapes = self.asset.root_physx_view.max_shapes
                if num_shapes != expected_shapes:
                    raise ValueError(
                        "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                        f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                    )
            else:
                raise ValueError("Rigid body materials are only supported for Articulation assets.")

        self.materials = asset.root_physx_view.get_material_properties().to(device=asset.device)
        body_ids_list = []
        if len(self.num_shapes_per_body) != 0:
            # sample material properties from the given ranges
            for body_ids in asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_ids])
                end_idx = start_idx + self.num_shapes_per_body[body_ids]
                for i in range(start_idx, end_idx):
                    body_ids_list.append(i)
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
            body_ids_tensor = torch.tensor(body_ids_list, device=asset.device)
        return self.materials[:, body_ids_tensor, :].flatten(start_dim=1)


rigid_body_materials = RigidBodyMaterialsWrapper()


def ee_target_error_with_latency(
    env: ManagerBasedRLEnv,
    command_name: str = "EE_pose",
    pos_scale: float = 1.0,
    orn_scale: float = 1.0,
) -> torch.Tensor:
    """
    Computes the observation of the EE pose error with latency.
    
    Returns:
        torch.Tensor: Concatenated tensor of [Relative Position (3), Relative Rotation 6D (6)]
                      Total dimension: 9
    """
    # 1. Get Command Term
    cmd_term = env.command_manager.get_term(command_name)
    
    # 2. Get Delayed EE Pose (in World Frame)
    if hasattr(cmd_term, "get_delayed_ee_pose"):
        delayed_ee_pose_w = cmd_term.get_delayed_ee_pose()
    else:
        # Fallback to current pose if not using PicklePoseSequenceCommand
        # body_idx is available in UniformPoseCommand
        body_idx = cmd_term.body_idx
        current_pos = cmd_term.robot.data.body_state_w[:, body_idx, :3]
        current_quat = cmd_term.robot.data.body_state_w[:, body_idx, 3:7]
        delayed_ee_pose_w = torch.cat([current_pos, current_quat], dim=-1)

    ee_pos_w = delayed_ee_pose_w[:, :3]
    ee_quat_w = delayed_ee_pose_w[:, 3:7]
    
    # 3. Get Target Pose (in World Frame)
    # Assuming UniformWorldPoseCommand or compatible structure
    target_pose_w = cmd_term.pose_command_w
    target_pos_w = target_pose_w[:, :3]
    target_quat_w = target_pose_w[:, 3:7]
    
    # 4. Compute Relative Pose (Target expressed in Delayed EE Frame)
    # Normalize quaternions to avoid numerical instability (and NaN)
    # Add epsilon to norm to prevent division by zero if quaternion is close to zero
    ee_quat_w = math_utils.normalize(ee_quat_w)
    target_quat_w = math_utils.normalize(target_quat_w)

    # Relative Position: R_ee_inv * (p_target - p_ee)
    rel_pos = math_utils.quat_rotate_inverse(ee_quat_w, target_pos_w - ee_pos_w)
    
    # Relative Rotation: q_ee_inv * q_target
    rel_quat = math_utils.quat_mul(math_utils.quat_inv(ee_quat_w), target_quat_w)
    
    # Normalize result quaternion as well
    rel_quat = math_utils.normalize(rel_quat)
    
    # 5. Format Observation
    # Using 6D rotation representation (first two columns of rotation matrix) + position
    rel_rot_mat = math_utils.matrix_from_quat(rel_quat)
    rot_6d = rel_rot_mat[:, :, :2].flatten(1) # (N, 6)
    
    # Apply scaling to match observation ranges (UMI-style)
    rel_pos = rel_pos * pos_scale
    rot_6d = rot_6d * orn_scale

    return torch.cat([rel_pos, rot_6d], dim=-1)
