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


def export_cn_policy_as_onnx(
    obs_dim, estimation_dim, actor_critic, path, normalizer=None, filename="policy.onnx", verbose=False
):
    os.makedirs(path, exist_ok=True)
    _OnnxCNPolicyExporter(obs_dim, estimation_dim, actor_critic, normalizer, verbose).export(path, filename)


class _OnnxCNPolicyExporter(torch.nn.Module):
    def __init__(self, obs_dim, estimation_dim, actor_critic, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.obs_dim = obs_dim
        self.estimation_dim = estimation_dim
        self.actor = copy.deepcopy(actor_critic.actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else torch.nn.Identity()

    def forward(self, obs, latent):
        return self.actor(torch.cat([self.normalizer(obs), latent], dim=-1))

    def export(self, path, filename):
        self.eval()
        self.to("cpu")
        out = os.path.join(path, filename)
        obs = torch.zeros(1, self.obs_dim)
        estimation = torch.zeros(1, self.estimation_dim)
        torch.onnx.export(
            self, (obs, estimation), out,
            export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
            input_names=["obs", "estimation"],
            output_names=["actions"],
            dynamic_axes={},
        )
        _patch_ir_version(out)


def export_contactNet_as_onnx(
    obs_dim, contactNet, path, normalizer=None, filename="contactNet.onnx", verbose=False
):
    os.makedirs(path, exist_ok=True)
    _OnnxContactNetExporter(obs_dim, contactNet, normalizer=normalizer, verbose=verbose).export(path, filename)


class _OnnxContactNetExporter(torch.nn.Module):
    def __init__(self, obs_dim, contactNet, batch_first=True, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.obs_dim = obs_dim
        self.contactNet = copy.deepcopy(contactNet)
        self.batch_first = batch_first
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward(self, obs_history: torch.Tensor):
        if self.batch_first:
            batch_size, seq_length, obs_dim = obs_history.shape
        assert obs_dim == self.obs_dim, f"Expected obs_dim={self.obs_dim}, but got {obs_dim}"
        normalized_obs = self.normalizer(obs_history.view(-1, obs_dim))
        if self.batch_first:
            normalized_obs = normalized_obs.view(batch_size, seq_length, obs_dim)
        else:
            normalized_obs = normalized_obs.view(seq_length, batch_size, obs_dim)
        self.contactNet.eval()
        return self.contactNet.forward(normalized_obs)

    def export(self, path, filename):
        self.eval()
        self.to("cpu")
        out = os.path.join(path, filename)
        example_input = torch.zeros(2, 10, self.obs_dim)
        torch.onnx.export(
            self, example_input, out,
            export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
            input_names=["obs_history"],
            output_names=["latent"],
            dynamic_axes={
                "obs_history": {0: "batch_size", 1: "sequence_length"},
                "latent": {0: "sequence_length", 1: "batch_size"},
            },
        )
        _patch_ir_version(out)
