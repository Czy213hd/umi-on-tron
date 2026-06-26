
import gymnasium as gym

from . import wf_tron1_arm_env_cfg
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-EEPose-Rough-Limx-WF-Tron1A-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wf_tron1_arm_env_cfg.LimxEEposeRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.implicit_one_stage_cfg:ImplicitOneStageRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-EEPose-Rough-Limx-WF-Tron1A-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wf_tron1_arm_env_cfg.LimxEEposeRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.implicit_one_stage_cfg:ImplicitOneStageRunnerCfg",
    },
)