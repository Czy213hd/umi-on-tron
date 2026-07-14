
import gymnasium as gym

from . import sf_tron1_arm_env_cfg
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sf_tron1_arm_env_cfg.LimxEEposeRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.implicit_one_stage_cfg:ImplicitOneStageRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sf_tron1_arm_env_cfg.LimxEEposeRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.implicit_one_stage_cfg:ImplicitOneStageRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-Command-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sf_tron1_arm_env_cfg.LimxEEposeCommandEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.implicit_one_stage_cfg:ImplicitOneStageRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-FPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sf_tron1_arm_env_cfg.LimxEEposeRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.fpo_one_stage_cfg:FpoOneStageRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-FPO-Command-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sf_tron1_arm_env_cfg.LimxEEposeCommandEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.fpo_one_stage_cfg:FpoOneStageRunnerCfg",
    },
)
gym.register(
    id="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-FPO-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": sf_tron1_arm_env_cfg.LimxEEposeRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.fpo_one_stage_cfg:FpoOneStageRunnerCfg",
    },
)
