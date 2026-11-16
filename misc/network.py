import torch
import torch.nn as nn

from typing import Iterable


@torch.no_grad()
def polyak_update(params: Iterable[nn.Parameter], target_params: Iterable[nn.Parameter], tau: float) -> None:
    """Soft-update the target parameters: target = (1 - tau) * target + tau * param."""
    for param, target_param in zip(params, target_params):
        if tau == 1:
            target_param.data.copy_(param.data)
        else:
            target_param.data.mul_(1.0 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def mlp(input_dim: int,
        output_dim: int,
        net_arch: list[int],
        activation_fn: type[nn.Module] = nn.ReLU,
        drop_rate: float = 0.0,
        layer_norm: bool = False) -> nn.Sequential:
    """Build an MLP with optional dropout and LayerNorm per hidden layer; output_dim <= 0 omits the output layer."""
    modules = []
    for in_dim, out_dim in zip([input_dim] + net_arch[:-1], net_arch):
        modules.append(nn.Linear(in_dim, out_dim))
        if drop_rate > 0.0:
            modules.append(nn.Dropout(p=drop_rate))
        if layer_norm:
            modules.append(nn.LayerNorm(out_dim))
        modules.append(activation_fn())
    if output_dim > 0:
        modules.append(nn.Linear(net_arch[-1], output_dim))
    return nn.Sequential(*modules)


@torch.no_grad()
def layer_init(layer: nn.Module, method: str = 'orthogonal', weight_gain: float = 1, bias_const: float = 0) -> None:
    """Initialize the weights of Linear/Conv2d layers (orthogonal or Xavier) and set a constant bias."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if method == 'xavier':
            nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif method == 'orthogonal':
            nn.init.orthogonal_(layer.weight, gain=weight_gain)
        nn.init.constant_(layer.bias, bias_const)
