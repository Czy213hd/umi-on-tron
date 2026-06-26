# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
# from isaaclab_tasks.utils.wrappers.rsl_rl import (
#     RslRlPpoActorCriticCfg,
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoAlgorithmCfg,
# )

from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from datetime import datetime
from dataclasses import MISSING


@configclass
class GruCfg:
    gru_latent_dim: int = MISSING
    gru_input_dim: int = MISSING
    gru_batch_first: bool = True


@configclass
class ContactNetCfg:
    # input_dim: int = MISSING
    output_dim: int = MISSING
    model_dim: int = MISSING
    num_layers: int = MISSING
    num_heads: int = MISSING
    dim_feedforward: int = MISSING
    next_obs_decoder_hidden_dims: list = MISSING
    next_obs_decoder_input_dim: int = MISSING
    dropout: float = MISSING
    class_name: str = "ContactNetModel"


@configclass
class PpoIOSCfg(RslRlPpoAlgorithmCfg):
    next_obs_latent_dim: int = MISSING
    beta: float = MISSING


@configclass
class ImplicitOneStageRunnerCfg(RslRlOnPolicyRunnerCfg):

    # run_name = "1.0"
    cn_obs_hist_len = 10
    # gru_update_interval = 4
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 200
    experiment_name = "ImplicitOneStageARXR5Arm"
    # experiment_name = "ImplicitOneStage"
    empirical_normalization = True
    # wandb settings
    wandb_project = "NYX-OneStageRFM"
    wandb_run_name = datetime.now().strftime("%m-%d_%H:%M")
    wandb_mode = "online"
    ppo_algorithm = PpoIOSCfg(
        next_obs_latent_dim=64,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.007,
        num_learning_epochs=5,
        num_mini_batches=4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        learning_rate=3.0e-4,
        beta=0.1,
    )
    ppo_algorithm.grad_coef = 0.0

    adaptive_entropy = {
        "is_used": True,
        "start_value": 0.015,
        "end_value": 0.0005,
        "start_point": 0,
        "end_point": 5000,
    }
    grad_penalty_scheme = {
        "is_used": True,
        "start_point": 0.0,
        "start_value": 0.0,
        "end_value": 0.0002,
        "end_point": 6000,
    }

    contactNet = ContactNetCfg(
        # output_dim=128,
        model_dim=128,
        num_layers=2,
        num_heads=8,
        dim_feedforward=512,
        next_obs_decoder_hidden_dims=[256, 128],
        dropout=0.0,
        class_name="SimplifiedContactNetModel",
    )
    contactNet.output_dim = 3 + 2 * ppo_algorithm.next_obs_latent_dim
    contactNet.next_obs_decoder_input_dim = ppo_algorithm.next_obs_latent_dim

    gru = GruCfg()
    gru.gru_latent_dim = contactNet.output_dim  # next_obs_latent, base_lin_vel, RL latent.
    gru.gru_input_dim = contactNet.output_dim

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )