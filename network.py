import math
from typing import Tuple, List, Any

import torch
import torch.nn as nn

from network_utils import mlp, conv3x3, support_to_scalar, scalar_to_support, normalize_hidden_state


class ConvBlock(nn.Module):
    """Convolutional block"""

    def __init__(self,
                in_channels: int,
                out_channels: int,
                stride: int=1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.conv3x3(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self,
                n_channels: int,
                stride: int=1) -> None:
        super().__init__()
        self.conv1 = conv3x3(n_channels, n_channels, stride)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = conv3x3(n_channels, n_channels)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class DownSample(nn.Module):
    """Network to reduce the spatial resolution of observations before representation network"""

    def __init__(self,
                in_channels: int,
                out_channels: int) -> None:
        # (C_in x h x w) -> (C_out/2 x h/2 x w/2)
        self.conv1 = conv3x3(in_channels, out_channels / 2, stride=2)

        # (C_out/2 x h/2 x w/2) -> (C_out/2 x h/2 x w/2)
        self.res_block1 = nn.Sequential([
            ResidualBlock(out_channels / 2) for _ in range(2)
        ])

        # (C_out/2 x h/2 x w/2) -> (C_out x h/4 x w/4)
        self.conv2 = conv3x3(out_channels / 2, out_channels, stride=2)

        # (C_out x h/4 x w/4) -> (C_out x h/4 x w/4)
        self.res_block2 = nn.Sequential([
            ResidualBlock(out_channels) for _ in range(3)
        ])

        # (C_out x h/4 x w/4) -> (C_out x h/8 x w/8)
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # (C_out x h/8 x w/8) -> (C_out x h/8 x w/8)
        self.res_block3 = nn.Sequential([
            ResidualBlock(out_channels) for _ in range(3)
        ])

        # (C_out x h/8 x w/8) -> (C_out x h/16 x w/16)
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.res_block1(out)
        out = self.conv2(out)
        out = self.res_block2(out)
        out = self.pooling1(out)
        out = self.res_block3(out)
        out = self.pooling2(out)
        return out


class RepresentationNetwork(nn.Module):
    """Representation network

    If downsample is turned on, assuming that observation is in 3D (n_colors x h x w)
    and input is (stacked observations + encoded action):
        - input: (n_stacked_observations * n_colors + n_stacked_observations) x h x w
        - output: downsampled_h x downsampled_w x n_channels
    Otherwise, observation has shape (h * w) and input has shape (n_stacked_observations x h x w)
    
    :param observation_dim: Observation dimensionality
    :param n_stacked_observations: Number of observations stacked
    :param n_blocks: Number of residual blocks
    :param n_channels: Number of channels in the ResNet
    :param downsample: Whether to use downsample on inputted observation
    """

    def __init__(self,
                observation_dim: List[int],
                n_stacked_observations: int,
                n_blocks: int,
                n_channels: int,
                downsample: bool) -> None:
        super().__init__()

        self.downsample = downsample
        if self.downsample:
            # C_in = n_stacked_observations * observation_dim[0] + n_stacked_observations
            # (C_in x h x w) -> (n_channels x h/16 x w/16)
            self.downsample_net = DownSample(
                n_stacked_observations * observation_dim[0] + n_stacked_observations,
                n_channels
            )
        else:
            # (n_stacked_observations x h x w) -> (n_channels x h x w)
            self.conv_block = ConvBlock(n_stacked_observations, n_channels)

        # preserves resolution
        self.res_tower = nn.Sequential([
            ResidualBlock(n_channels) for _ in range(n_blocks)
        ])


    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            out = self.downsample_net(observation)
        else:
            out = self.conv_block(observation)
        hidden_state = self.res_tower(out)
        return hidden_state


class DynamicsNetwork(nn.Module):
    """Dynamics network"""

    def __init__(self,
                n_blocks: int,
                n_channels: int,
                reduced_channels_reward: int,
                fc_reward_layers: List[int],
                support_size: int,
                block_output_size_reward: int) -> None:
        super().__init__()
        self.conv_block = ConvBlock(n_channels, n_channels - 1)
        self.res_tower = nn.Sequential([
            ResidualBlock(n_channels - 1) for _ in range(n_blocks)
        ])

        self.reward_head = nn.Sequential(
            nn.Conv2d(n_channels - 1, reduced_channels_reward, stride=1),
            nn.Flatten(start_dim=1),
            mlp([block_output_size_reward, *fc_reward_layers, support_size])
        )


    def forward(self, state_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.conv_block(state_action)
        out = self.res_tower(out)
        next_hidden_state = out
        reward = self.reward_head(out)
        return next_hidden_state, reward


class PredictionNetwork(nn.Module):
    """Prediction network"""

    def __init__(self,
                n_blocks: int,
                n_channels: int,
                reduced_channels_policy: int,
                reduced_channels_value: int,
                fc_policy_layers: int,
                fc_value_layers: int,
                support_size: int,
                action_space_size: int,
                block_output_size_policy: int,
                block_output_size_value: int) -> None:
        super().__init__()
        self.res_tower = nn.Sequential([
            ResidualBlock(n_channels) for _ in range(n_blocks)
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(n_channels, reduced_channels_policy, 1),
            nn.Flatten(start_dim=1),
            mlp([block_output_size_policy, *fc_policy_layers, support_size]),
            nn.Softmax(dim=0)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(n_channels, reduced_channels_value, 1),
            nn.Flatten(start_dim=1),
            mlp([block_output_size_value, *fc_value_layers, action_space_size])
        )


    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.res_tower(hidden_state)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


class MuZeroNetwork(nn.Module):

    def __init__(self,
                observation_dim: List[int],
                n_stacked_observations: int,
                n_blocks: int,
                n_channels: int,
                downsample: bool,
                reduced_channels_reward: int,
                reduced_channels_policy: int,
                reduced_channels_value: int,
                fc_reward_layers: int,
                fc_policy_layers: int,
                fc_value_layers: int,
                support_limit: int,
                action_space_size: int) -> None:
        super().__init__()
        self.support_limit = support_limit
        support_size = self.support_limit * 2 + 1
        if downsample:
            downsampled_res = math.ceil(observation_dim[1] / 16) * math.ceil(observation_dim[2] / 16)
            block_output_size_reward = reduced_channels_reward * downsampled_res
            block_output_size_policy = reduced_channels_policy * downsampled_res
            block_output_size_value = reduced_channels_value * downsampled_res
        else:
            res = observation_dim[1] * observation_dim[2]
            block_output_size_reward = reduced_channels_reward * res
            block_output_size_policy = reduced_channels_policy * res
            block_output_size_value = reduced_channels_value * res
        
        self.repretation_network = RepresentationNetwork(
            observation_dim, n_stacked_observations, n_blocks, n_channels, downsample
        )
        self.dynamics_network = DynamicsNetwork(
            n_blocks, n_channels, reduced_channels_reward, fc_reward_layers, support_size, block_output_size_reward
        )
        self.prediction_network = PredictionNetwork(
            n_blocks, n_channels, reduced_channels_policy, reduced_channels_value, fc_policy_layers,
            fc_value_layers, support_size, action_space_size, block_output_size_policy, block_output_size_value
        )


    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        hidden_state = self.repretation_network(observation)
        return normalize_hidden_state(hidden_state)


    def dynamics(self,
                hidden_state: torch.Tensor,
                encoded_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_action = torch.cat((hidden_state, encoded_action), dim=1)
        next_hidden_state, reward = self.dynamics_network(state_action)
        return normalize_hidden_state(next_hidden_state), reward


    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy, value = self.prediction_network(hidden_state)
        return policy, value


    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """representation + prediction function"""
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)
        value = support_to_scalar(value, self.support_limit)
        return policy, hidden_state, value


    def recurrent_inference(self,
                            hidden_state: torch.Tensor,
                            encoded_action: torch.Tensor) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
        """dynamics + prediction function"""
        next_hidden_state, reward = self.dynamics(hidden_state, encoded_action)
        policy, value = self.prediction(hidden_state)
        reward = support_to_scalar(reward)
        value = support_to_scalar(value)
        return policy, next_hidden_state, value, reward


    def set_weights(self, weights: Any) -> None:
        self.load_state_dict(weights)
