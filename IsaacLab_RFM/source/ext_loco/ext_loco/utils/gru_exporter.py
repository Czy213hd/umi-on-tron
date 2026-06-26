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


def export_gru_as_onnx(gru_wrapper, path, filename="gru.onnx", verbose=False):
    os.makedirs(path, exist_ok=True)
    _OnnxGRUExporter(gru_wrapper, verbose).export(path, filename)


class _OnnxGRUExporter(torch.nn.Module):
    def __init__(self, gru_wrapper, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.gru = copy.deepcopy(gru_wrapper)
        self.gru_latent_dim = gru_wrapper.gru_latent_dim
        self.input_dim = gru_wrapper.gru.input_size

    def forward(self, obs, hidden_state):
        # Export through a pure forward path to avoid mutating module attributes.
        output, new_hidden_state = self.gru.gru_forward_without_memory_with_hidden(obs, hidden_state)
        return output, new_hidden_state

    def export(self, path, filename):
        self.eval()
        self.to("cpu")
        out = os.path.join(path, filename)
        obs = torch.zeros(1, self.input_dim)
        hidden_state = torch.zeros(1, 1, self.gru_latent_dim)
        torch.onnx.export(
            self, (obs, hidden_state), out,
            export_params=True, opset_version=15, dynamo=False, verbose=self.verbose,
            input_names=["cn_output", "hidden_state"],
            output_names=["gru_latent", "new_hidden_state"],
            dynamic_axes={
                "cn_output": {0: "batch_size"},
                "hidden_state": {1: "batch_size"},
                "gru_latent": {0: "batch_size"},
                "new_hidden_state": {1: "batch_size"},
            },
        )
        _patch_ir_version(out)
