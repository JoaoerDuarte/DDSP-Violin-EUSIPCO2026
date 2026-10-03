import math

import torch
import torch.nn as nn

from .model_utils import HARMONIC_DISTRIBUTION, Z


def mlp(input_size: int, hidden_size: int, n_layers: int) -> nn.Sequential:
    """n_layers blocks of Linear, LayerNorm and LeakyReLU."""
    sizes = [input_size] + n_layers * [hidden_size]
    layers = []
    for i in range(n_layers):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.LayerNorm(sizes[i + 1]), nn.LeakyReLU()]
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    """Map frame-wise inputs to named synthesis parameters.

    Each input goes through its own MLP and a GRU runs over their concatenation. An MLP
    and a linear layer then map the GRU output, together with the input MLP outputs, to
    the parameters listed in outputs as (name, size).
    """

    def __init__(self, inputs: list[str], outputs: list[tuple[str, int]], z_dims: int,
                 rnn_channels: int, ch: int, layers_per_stack: int):
        super().__init__()
        self.input_names = inputs
        self.output_sizes = outputs
        self.input_stacks = nn.ModuleDict({
            name: mlp(z_dims if name == Z else 1, ch, layers_per_stack) for name in inputs
        })
        self.gru = nn.GRU(len(inputs) * ch, rnn_channels, batch_first=True)
        self.out_stack = mlp(rnn_channels + len(inputs) * ch, ch, layers_per_stack)
        self.dense_out = nn.Linear(ch, sum(size for _, size in outputs))

    def init_harmonic_bias_1_over_n(self):
        """Set the harmonic-distribution biases so that exp_sigmoid, with its default
        parameters, maps them to 1/n for harmonic n."""
        start = 0
        for name, size in self.output_sizes:
            if name == HARMONIC_DISTRIBUTION:
                break
            start += size
        log_exp, max_value, threshold = math.log(10.0), 2.0, 1e-7
        bias = torch.zeros(size)
        for i in range(size):
            sigmoid_value = ((1.0 / (i + 1) - threshold) / max_value) ** (1.0 / log_exp)
            sigmoid_value = max(min(sigmoid_value, 1.0 - 1e-6), 1e-6)
            bias[i] = math.log(sigmoid_value / (1.0 - sigmoid_value))
        self.dense_out.bias.data[start:start + size] = bias

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """inputs: [batch, frames, channels] each. Returns raw parameters [batch, frames, size]."""
        hidden = [self.input_stacks[name](inputs[name]) for name in self.input_names]
        x = self.gru(torch.cat(hidden, dim=-1))[0]
        x = self.dense_out(self.out_stack(torch.cat([x] + hidden, dim=-1)))
        params, start = {}, 0
        for name, size in self.output_sizes:
            params[name] = x[..., start:start + size]
            start += size
        return params
