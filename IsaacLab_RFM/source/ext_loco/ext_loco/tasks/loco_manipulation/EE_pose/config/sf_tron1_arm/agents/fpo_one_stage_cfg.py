# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .implicit_one_stage_cfg import ContactNetCfg, GruCfg, ImplicitOneStageRunnerCfg, PpoIOSCfg


@configclass
class FpoOneStageRunnerCfg(ImplicitOneStageRunnerCfg):
    experiment_name = "ImplicitOneStageARXR5ArmFPO"

    ppo_algorithm = PpoIOSCfg(
        next_obs_latent_dim=64,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        learning_rate=3.0e-4,
        beta=0.1,
        flow_matching=True,
        parameterization="velocity",
        solver_step_size=0.1,
        prior_noise_std=1.0,
        perturb_action_std=0.03,
        sample_t_strategy="uniform",
        p_mean=-1.2,
        p_std=1.2,
        zero_action_input=False,
        condition_drop_ratio=0.0,
    )
    ppo_algorithm.grad_coef = 0.0

    contactNet = ContactNetCfg(
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
    gru.gru_latent_dim = contactNet.output_dim
    gru.gru_input_dim = contactNet.output_dim

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="silu",
    )
    policy.class_name = "FlowActorCritic"
