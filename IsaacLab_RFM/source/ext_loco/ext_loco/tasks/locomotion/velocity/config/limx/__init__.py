import gymnasium as gym

from . import agents,limx_velocity_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-Velocity-Rough-Limx-WF-Tron1A-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_velocity_env_cfg.LimxLocomotionVelocityRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LimxPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Rough-Limx-WF-Tron1A-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_velocity_env_cfg.LimxLocomotionVelocityRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LimxPPORunnerCfg",
    },
)

