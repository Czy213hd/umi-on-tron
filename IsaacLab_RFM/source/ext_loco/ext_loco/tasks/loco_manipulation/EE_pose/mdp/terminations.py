from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


class TimeOutStochasticWrapper:
    """
    A wrapper class for calculating stochastic time_out condition.
    """

    def __init__(self):
        self.max_episode_length = None
        self.__name__ = "time_out_stochastic"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        if self.max_episode_length is None:
            self.max_episode_length = (
                torch.ones(env.episode_length_buf.shape, device=env.episode_length_buf.device) * env.max_episode_length
            )
            random_values = torch.rand(self.max_episode_length.shape, device=self.max_episode_length.device)
            random_scale = random_values + 0.5
            self.max_episode_length *= random_scale

        time_out = env.episode_length_buf >= self.max_episode_length

        # resample max_episode_length with stochastic mask
        time_out_ids = time_out.nonzero(as_tuple=False).squeeze(-1)
        if len(time_out_ids) == 0:
            return time_out
        random_values = torch.rand(time_out_ids.shape, device=time_out_ids.device)
        random_scale = random_values + 0.5
        self.max_episode_length[time_out_ids] = env.max_episode_length * random_scale
        return time_out


time_out_stochastic = TimeOutStochasticWrapper()


"""
Fail terminations.
"""


def bad_orientation_stochastic(
    env: ManagerBasedRLEnv, limit_angle: float, probability: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    bad_orientation = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
    random_values = torch.rand(bad_orientation.shape, device=bad_orientation.device)
    return bad_orientation & (random_values < probability)


def bad_height_stochastic(
    env: ManagerBasedRLEnv, limit_height: float, probability: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    bad_height = asset.data.root_link_pos_w[:, 2].abs() < limit_height
    random_values = torch.rand(bad_height.shape, device=bad_height.device)
    return bad_height & (random_values < probability)


def illegal_contact_stochastic(
    env: ManagerBasedRLEnv, threshold: float, probability: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    illegal_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    random_values = torch.rand(illegal_contact.shape, device=illegal_contact.device)
    return illegal_contact & (random_values < probability)


def overcontact_stochastic(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float, probability: float
) -> torch.Tensor:
    """Terminate when the number of contacts exceeds the threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    over_contact = (
        torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1)[0]
        > threshold
    ).any(dim=-1)
    # over_contact = (
    #     torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    #     > threshold
    # ).any(dim=-1)
    random_values = torch.rand(over_contact.shape, device=over_contact.device)
    return over_contact & (random_values < probability)