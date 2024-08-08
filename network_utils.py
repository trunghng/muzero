from typing import List, Callable, Dict

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes: List[int],
        activation: nn.Module=nn.ELU,
        output_activation: nn.Module=nn.Identity):
    """
    Create an MLP

    :param sizes: List of layers' size
    :param activation: Activation layer type
    :param output_activation: Output layer type
    """
    layers = []
    for i in range(len(sizes) - 1):
        activation_ = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation_()]
    return nn.Sequential(*layers)


def conv3x3(in_channels: int, out_channels: int, stride: int=1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def atari_scalar_transform(x: torch.Tensor, var_eps: float=0.001) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + var_eps * x


def inv_atari_scalar_transform(x: torch.Tensor, var_eps: float=0.001) -> torch.Tensor:
    return torch.sign(x) * (((torch.sqrt(1 + 4 * var_eps * (torch.abs(x)
                + 1 + var_eps)) - 1) / (2 * var_eps)) ** 2 - 1)


def support_to_scalar(probabilities: torch.Tensor,
                      support_limit: int,
                      inv_scalar_transformer: Callable=inv_atari_scalar_transform,
                      **kwargs) -> torch.Tensor:
    """
    Re-convert categorical representation of scalars with integer support [-support_limit, support_limit] back to scalars

    :param probabilities: Tensor represents categorical distributions
    :param support_limit: Number of categories indicating range symmetric around 0
    :param inv_scalar_transformer: Inverse of the function that scaled scalars before converting to distributions
    :param kwargs: Keyword arguments for inv_scalar_transformer
    """
    support = torch.arange(-support_limit, support_limit + 1)
    x = np.dot(probabilities.detach().cpu(), support)
    x = inv_scalar_transformer(torch.as_tensor(x, device=probabilities.device), **kwargs)
    return x


def scalar_to_support(x: torch.Tensor,
                      support_limit: int,
                      scalar_transformer: Callable=atari_scalar_transform,
                      **kwargs) -> torch.Tensor:
    """
    Convert scalars to categorical representations with integer support [-support_limit, support_limit]

    :param x: Tensor of scalars
    :param support_limit: Number of categories indicating range symmetric around 0
    :param scalar_transformer: Function to scale scalars before conversion
    :param kwargs: Keyword arguments for scalar_transformer
    """
    x = scalar_transformer(x, **kwargs)
    x = torch.clamp(x, min=-support_limit, max=support_limit)
    floor = x.floor().int()
    prob = x - floor
    probabilities = torch.zeros((x.shape[0], x.shape[1], support_limit * 2 + 1), device=x.device)
    probabilities[:, :, floor + support_limit] = 1 - prob
    probabilities[:, :, floor + support_limit + 1] = prob
    return probabilities


def normalize_hidden_state(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Scale hidden states to [0, 1]:
        s_scaled = (s - min(s)) / (max(s) - min(s))

    :param hidden_states: batch of hidden states
    """
    # Scale hidden state into [0, 1], performed strictly over one example, not batch
    hidden_states_ = hidden_states.view(-1, hidden_states.shape[1], hidden_states.shape[2] * hidden_states.shape[3])
    # Also insert one dim to make broadcasting valid
    hidden_states_max = torch.max(hidden_states_, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    hidden_states_min = torch.min(hidden_states_, dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    hidden_states_scaled = (hidden_states - hidden_states_min) / (hidden_states_max - hidden_states_min + 1e-5)
    return hidden_states_scaled


def scale_gradient(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Scale gradients for reverse differentiation proportional to the given scale"""
    x.register_hook(lambda grad: grad * scale_factor)


def dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cpu_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict
