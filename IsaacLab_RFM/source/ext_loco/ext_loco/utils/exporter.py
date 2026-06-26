import copy
import os
import torch
import onnx


def _patch_ir_version(filepath, target=8):
    model = onnx.load(filepath)
    if model.ir_version > target:
        model.ir_version = target
    onnx.save_model(model, filepath, save_as_external_data=False)
    data_file = filepath + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)


def export_actor_as_onnx(
    actor_critic, obs_dim, path, normalizer=None, filename="policy.onnx", verbose=False
):
    os.makedirs(path, exist_ok=True)
    _OnnxActorExporter(actor_critic, obs_dim, normalizer, verbose).export(path, filename)


class _OnnxActorExporter(torch.nn.Module):
    def __init__(self, actor_critic, obs_dim, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_flow_matching = getattr(actor_critic, "is_flow_matching", False)
        self.actor_critic = copy.deepcopy(actor_critic) if self.is_flow_matching else None
        self.actor = copy.deepcopy(actor_critic.actor) if not self.is_flow_matching else None
        self.is_recurrent = actor_critic.is_recurrent
        self.obs_dim = obs_dim
        self.num_actor_obs = actor_critic.num_actor_obs
        self.num_actions = actor_critic.num_actions
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        pass

    def forward(self, obs, latent, noise=None, condition_mask=None):
        normalized_obs = self.normalizer(obs)
        if self.is_flow_matching:
            if noise is None:
                noise = torch.zeros(obs.shape[0], self.num_actions, device=obs.device, dtype=obs.dtype)
            return self.actor_critic.export_actor(normalized_obs, latent, noise, condition_mask=condition_mask)
        return self.actor(torch.cat([normalized_obs, latent], dim=-1))

    def export(self, path, filename):
        self.eval()
        self.to("cpu")
        out = os.path.join(path, filename)
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            torch.onnx.export(
                self, (obs, h_in, c_in), out,
                export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.obs_dim)
            latent = torch.zeros(1, self.num_actor_obs - self.obs_dim)
            if self.is_flow_matching:
                noise = torch.zeros(1, self.num_actions)
                condition_mask = torch.ones(1, self.num_actor_obs)
                torch.onnx.export(
                    self, (obs, latent, noise, condition_mask), out,
                    export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
                    input_names=["obs", "next_gru_latent", "noise", "condition_mask"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
            else:
                torch.onnx.export(
                    self, (obs, latent), out,
                    export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
                    input_names=["obs", "next_gru_latent"],
                    output_names=["actions"],
                    dynamic_axes={},
                )
        _patch_ir_version(out)


def export_ts_policy_as_onnx(
    obs_dim, latent_dim, actor_critic, path, normalizer=None, filename="policy.onnx", verbose=False
):
    os.makedirs(path, exist_ok=True)
    _OnnxTSPolicyExporter(obs_dim, latent_dim, actor_critic, normalizer, verbose).export(path, filename)


class _OnnxTSPolicyExporter(torch.nn.Module):
    def __init__(self, obs_dim, latent_dim, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.forward = self.forward_lstm
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        return self.actor(x.squeeze(0)), h, c

    def forward(self, obs, latent):
        return self.actor(torch.cat([self.normalizer(obs), latent], dim=-1))

    def export(self, path, filename):
        self.eval()
        self.to("cpu")
        out = os.path.join(path, filename)
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            torch.onnx.export(
                self, (obs, h_in, c_in), out,
                export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.obs_dim)
            latent = torch.zeros(1, self.latent_dim)
            torch.onnx.export(
                self, (obs, latent), out,
                export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
                input_names=["obs", "latent"],
                output_names=["actions"],
                dynamic_axes={},
            )
        _patch_ir_version(out)


def export_ts_encoder_as_onnx(
    obs_dim, encoder, path, normalizer=None, filename="encoder.onnx", verbose=False
):
    os.makedirs(path, exist_ok=True)
    _OnnxTSEncoderExporter(obs_dim, encoder, normalizer, verbose).export(path, filename)


class _OnnxTSEncoderExporter(torch.nn.Module):
    def __init__(self, obs_dim, encoder, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.obs_dim = obs_dim
        self.encoder = copy.deepcopy(encoder)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward(self, obs_history):
        return self.encoder(self.normalizer(obs_history.view(-1, self.obs_dim)).view(1, -1))

    def export(self, path, filename):
        self.eval()
        self.to("cpu")
        out = os.path.join(path, filename)
        obs_history = torch.zeros(1, self.encoder[0].in_features)
        torch.onnx.export(
            self, obs_history, out,
            export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
            input_names=["obs_history"],
            output_names=["latent"],
            dynamic_axes={},
        )
        _patch_ir_version(out)
