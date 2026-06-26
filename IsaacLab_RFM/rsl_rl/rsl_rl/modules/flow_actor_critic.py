from __future__ import annotations

import math

import torch
import torch.nn as nn

from .actor_critic import get_activation


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=max(half, 1), dtype=torch.float32, device=t.device) / max(half, 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class CondOTPath:
    @staticmethod
    def sample(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_expanded = t.reshape(-1, *([1] * (x_1.ndim - 1)))
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * x_1
        dx_t = x_1 - x_0
        return x_t, dx_t

    @staticmethod
    def target_to_velocity(x_1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_expanded = t.reshape(-1, *([1] * (x_t.ndim - 1)))
        return (x_1 - x_t) / torch.clamp(1.0 - t_expanded, min=1.0e-6)

    @staticmethod
    def velocity_to_target(velocity: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_expanded = t.reshape(-1, *([1] * (x_t.ndim - 1)))
        return x_t + (1.0 - t_expanded) * velocity


class FlowActor(nn.Module):
    TASK_POLICY_OBS_DIM = 65
    COMMAND_POS_SLICE = slice(6, 9)
    COMMAND_ROT_SLICE = slice(9, 15)
    COMMAND_DISTANCE_SLICE = slice(64, 65)

    def __init__(
        self,
        num_actor_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int],
        activation: str,
        parameterization: str = "velocity",
        solver_step_size: float = 0.1,
        prior_noise_std: float = 1.0,
        perturb_action_std: float = 0.03,
        sample_t_strategy: str = "uniform",
        logprob_std: float = 0.05,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        zero_action_input: bool = False,
        condition_drop_ratio: float = 0.0,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        if parameterization not in {"velocity", "data"}:
            raise ValueError(f"Unsupported parameterization: {parameterization}")
        if sample_t_strategy not in {"uniform", "lognormal"}:
            raise ValueError(f"Unsupported sample_t_strategy: {sample_t_strategy}")

        flow_steps = round(1.0 / solver_step_size)
        if flow_steps <= 0 or not math.isclose(flow_steps * solver_step_size, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("solver_step_size must evenly divide 1.0")

        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.parameterization = parameterization
        self.solver_step_size = solver_step_size
        self.flow_steps = int(flow_steps)
        self.prior_noise_std = prior_noise_std
        self.perturb_action_std = perturb_action_std
        self.sample_t_strategy = sample_t_strategy
        self.logprob_std = logprob_std
        self.p_mean = p_mean
        self.p_std = p_std
        self.zero_action_input = zero_action_input
        self.condition_drop_ratio = condition_drop_ratio
        self.path = CondOTPath()

        hidden_size = actor_hidden_dims[-1]
        feature_layers: list[nn.Module] = []
        input_dim = num_actor_obs + num_actions
        last_dim = input_dim
        for hidden_dim in actor_hidden_dims:
            feature_layers.append(nn.Linear(last_dim, hidden_dim))
            feature_layers.append(get_activation(activation))
            last_dim = hidden_dim
        self.actor_mlp = nn.Sequential(*feature_layers)
        self.noise_emb = TimestepEmbedder(hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.actor_norm = AdaLayerNorm(hidden_size)
        self.post_adaln_non_linearity = nn.SiLU()
        self.action_head = nn.Linear(hidden_size, num_actions)
        nn.init.normal_(self.action_head.weight, std=0.01)
        nn.init.zeros_(self.action_head.bias)

    def _sample_noise(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
        deterministic: bool,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is not None:
            return noise.to(device=device, dtype=dtype)
        if deterministic:
            return torch.zeros(shape, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype) * self.prior_noise_std

    def _sample_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.sample_t_strategy == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype)
        rnd_normal = torch.randn((batch_size,), device=device, dtype=dtype)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        time = 1.0 / (1.0 + sigma)
        return torch.clamp(time, min=1.0e-4, max=1.0)

    def _sample_condition_mask(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        condition_mask: torch.Tensor | None = None,
        use_condition_dropout: bool = True,
    ) -> torch.Tensor | None:
        if condition_mask is not None:
            return condition_mask.to(device=device, dtype=dtype)
        if not use_condition_dropout or self.condition_drop_ratio <= 0.0:
            return None

        mask = torch.ones(batch_size, self.num_actor_obs, device=device, dtype=dtype)
        if self.num_actor_obs < self.TASK_POLICY_OBS_DIM:
            return mask

        keep_prob = 1.0 - self.condition_drop_ratio
        segments = (
            self.COMMAND_POS_SLICE,
            self.COMMAND_ROT_SLICE,
            self.COMMAND_DISTANCE_SLICE,
        )
        for segment in segments:
            keep = torch.bernoulli(torch.full((batch_size, 1), keep_prob, device=device, dtype=dtype))
            mask[:, segment] = keep
        return mask

    def forward_velocity(
        self,
        actor_input: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_t_eff = torch.zeros_like(x_t) if self.zero_action_input else x_t
        actor_input_eff = actor_input if condition_mask is None else actor_input * condition_mask
        net_input = torch.cat([actor_input_eff, x_t_eff], dim=-1)
        hidden = self.actor_mlp(net_input)
        hidden = self.actor_norm(hidden, self.noise_emb(t * (0.0 if self.zero_action_input else 1.0)))
        hidden = self.post_adaln_non_linearity(hidden)
        pred = self.action_head(hidden)
        if self.parameterization == "velocity":
            velocity = pred
            x_1 = self.path.velocity_to_target(velocity=velocity, x_t=x_t, t=t)
        else:
            x_1 = pred
            velocity = self.path.target_to_velocity(x_1=x_1, x_t=x_t, t=t)
        return velocity, x_1

    def _integrate(
        self,
        actor_input: torch.Tensor,
        noise: torch.Tensor,
        deterministic: bool,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_t = noise
        step_size = actor_input.new_tensor(self.solver_step_size)
        for step in range(self.flow_steps):
            t_value = step / self.flow_steps
            t = actor_input.new_full((actor_input.shape[0],), t_value)
            velocity, _ = self.forward_velocity(actor_input, x_t, t, condition_mask=condition_mask)
            x_t = x_t + step_size * velocity
        if not deterministic and self.perturb_action_std > 0.0:
            x_t = x_t + torch.randn_like(x_t) * self.perturb_action_std
        return x_t

    def sample(
        self,
        actor_input: torch.Tensor,
        deterministic: bool = False,
        noise: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        use_condition_dropout: bool = True,
    ) -> torch.Tensor:
        sampled_noise = self._sample_noise(
            actor_input.shape[:-1] + (self.num_actions,),
            actor_input.device,
            actor_input.dtype,
            deterministic=deterministic,
            noise=noise,
        )
        sampled_condition_mask = self._sample_condition_mask(
            actor_input.shape[0],
            actor_input.device,
            actor_input.dtype,
            condition_mask=condition_mask,
            use_condition_dropout=use_condition_dropout,
        )
        return self._integrate(actor_input, sampled_noise, deterministic, condition_mask=sampled_condition_mask)

    def forward(
        self,
        actor_input: torch.Tensor,
        noise: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.sample(
            actor_input,
            deterministic=noise is None,
            noise=noise,
            condition_mask=condition_mask,
            use_condition_dropout=noise is None,
        )

    def flow_matching_loss(
        self,
        actions: torch.Tensor,
        actor_input: torch.Tensor,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        use_condition_dropout: bool = True,
        return_noise_t: bool = False,
    ):
        if noise is None:
            noise = self._sample_noise(actions.shape, actions.device, actions.dtype, deterministic=False)
        if t is None:
            t = self._sample_t(actions.shape[0], actions.device, actions.dtype)
        sampled_condition_mask = self._sample_condition_mask(
            actor_input.shape[0],
            actor_input.device,
            actor_input.dtype,
            condition_mask=condition_mask,
            use_condition_dropout=use_condition_dropout,
        )

        x_t, u_t = self.path.sample(x_0=noise, x_1=actions, t=t)
        predicted_velocity, predicted_target = self.forward_velocity(actor_input, x_t, t, condition_mask=sampled_condition_mask)
        if self.parameterization == "velocity":
            log_probs = -((predicted_velocity - u_t) ** 2) / (2 * self.logprob_std**2)
        else:
            log_probs = -((predicted_target - actions) ** 2) / (2 * self.logprob_std**2)
        log_probs = log_probs.mean(dim=-1)
        loss = -log_probs.mean()

        if return_noise_t:
            return log_probs, loss, noise, t, sampled_condition_mask, predicted_target
        return log_probs, loss, predicted_target


class FlowActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="silu",
        init_noise_std=1.0,
        parameterization="velocity",
        solver_step_size=0.1,
        prior_noise_std=1.0,
        perturb_action_std=0.03,
        sample_t_strategy="uniform",
        p_mean=-1.2,
        p_std=1.2,
        zero_action_input=False,
        condition_drop_ratio=0.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "FlowActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.is_flow_matching = True
        self.mean_bound_loss = torch.tensor(0.0)

        self.actor = FlowActor(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            parameterization=parameterization,
            solver_step_size=solver_step_size,
            prior_noise_std=prior_noise_std,
            perturb_action_std=perturb_action_std,
            sample_t_strategy=sample_t_strategy,
            p_mean=p_mean,
            p_std=p_std,
            zero_action_input=zero_action_input,
            condition_drop_ratio=condition_drop_ratio,
        )

        critic_activation = get_activation(activation)
        critic_layers: list[nn.Module] = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), critic_activation]
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(get_activation(activation))
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=False)
        self._last_action_mean = None
        self._last_action_std = None
        self._last_entropy = None

    @property
    def action_mean(self):
        return self._last_action_mean

    @property
    def action_std(self):
        return self._last_action_std

    @property
    def entropy(self):
        return self._last_entropy

    def reset(self, dones=None):
        pass

    def bound_loss(self, mu: torch.Tensor) -> torch.Tensor:
        soft_bound = 0.9
        mu_loss = torch.zeros_like(mu)
        mu_loss = torch.where(mu > soft_bound, (mu - soft_bound) ** 2, mu_loss)
        mu_loss = torch.where(mu < -soft_bound, (mu + soft_bound) ** 2, mu_loss)
        return mu_loss.mean()

    def _set_action_stats(self, observations: torch.Tensor, actions: torch.Tensor):
        self._last_action_mean = self.actor.sample(observations, deterministic=True).detach()
        self._last_action_std = torch.zeros_like(actions)
        self._last_entropy = torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    def act(self, observations: torch.Tensor):
        actions = self.actor.sample(observations, deterministic=False, use_condition_dropout=True)
        self._set_action_stats(observations, actions)
        return actions

    def get_actions_log_prob(self, actions):
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    def act_inference(self, observations):
        return self.actor.sample(observations, deterministic=False, use_condition_dropout=True)

    def export_actor(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        noise: torch.Tensor,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actor_input = torch.cat([obs, latent], dim=-1)
        return self.actor.sample(
            actor_input,
            deterministic=False,
            noise=noise,
            condition_mask=condition_mask,
            use_condition_dropout=False,
        )

    def evaluate(self, critic_observations):
        return self.critic(critic_observations)

    def flow_matching_loss(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        use_condition_dropout: bool = True,
        return_noise_t: bool = False,
    ):
        output = self.actor.flow_matching_loss(
            actions,
            observations,
            noise=noise,
            t=t,
            condition_mask=condition_mask,
            use_condition_dropout=use_condition_dropout,
            return_noise_t=return_noise_t,
        )
        if return_noise_t:
            log_probs, loss, sampled_noise, sampled_t, sampled_condition_mask, predicted_target = output
            self.mean_bound_loss = self.bound_loss(predicted_target)
            return log_probs, loss, sampled_noise, sampled_t, sampled_condition_mask
        log_probs, loss, predicted_target = output
        self.mean_bound_loss = self.bound_loss(predicted_target)
        return log_probs, loss
