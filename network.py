import math
from typing import Tuple, List, Any

import torch
import torch.nn as nn

from utils.network_utils import mlp, conv3x3, ConvBlock, ResidualBlock,\
    support_to_scalar, normalize_hidden_state, mlp_normalize_hidden_state


class MuZeroNetwork:

    def __new__(cls, config):
        if config.network == 'resnet':
            return MuZeroResidualNetwork(
                config.observation_dim,
                config.n_actions,
                config.n_stacked_observations,
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_policy,
                config.reduced_channels_value,
                config.resnet_fc_reward_layers,
                config.resnet_fc_policy_layers,
                config.resnet_fc_value_layers,
                config.downsample,
                config.support_limit
            )
        else:
            return MuZeroFullyConnectedNetwork(
                config.observation_dim,
                config.n_actions,
                config.n_stacked_observations,
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_policy_layers,
                config.fc_value_layers,
                config.fc_representation_layers,
                config.fc_hidden_state_layers,
                config.support_limit
            )


class BaseNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        if isinstance(self, MuZeroResidualNetwork):
            self.normalize_hidden_state = normalize_hidden_state
        else:
            self.normalize_hidden_state = mlp_normalize_hidden_state

    def representation(self, observation: torch.Tensor) -> torch.Tensor:
        """
        :param observation:     (B x (n_stacked_obs * obs_dim[0]) x h x w)
        :return hidden_state:   (B x channels x h/16 x w/16) | (B x channels x h x w)
        """
        hidden_state = self.representation_network(observation)
        return self.normalize_hidden_state(hidden_state)

    def dynamics(self,
                 hidden_state: torch.Tensor,
                 encoded_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param hidden_state:        (B x channels x h/16 x w/16) | (B x channels x h x w)
        :param encoded_action:      (B x h/16 x w/16) | (B x h x w) | (B x n_actions)
        :return next_hidden_state:  (B x channels x h/16 x w/16) | (B x channels x h x w)
        :return reward_logits:      (B x support_size)
        """
        if isinstance(self, MuZeroResidualNetwork):
            encoded_action = encoded_action.unsqueeze(1)
        state_action = torch.cat((hidden_state, encoded_action), dim=1)
        next_hidden_state, reward_logits = self.dynamics_network(state_action)
        return self.normalize_hidden_state(next_hidden_state), reward_logits

    def prediction(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param hidden_state:    (B x channels x h/16 x w/16) | (B x channels x h x w)
        :return policy_logits:  (B x n_actions)
        :return value_logits:   (B x support_size)
        """
        policy_logits, value_logits = self.prediction_network(hidden_state)
        return policy_logits, value_logits

    def initial_inference(self,
                          observation: torch.Tensor,
                          inv_transform: bool=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Representation + Prediction function

        :param observation:     (B x (n_stacked_obs * obs_dim[0]) x h x w)
        :param inv_transform: whether to apply inverse transformation on categorical form of value
        :return policy_logits:  (B x n_actions)
        :return hidden_state:   (B x channels x h/16 x w/16) | (B x channels x h x w)
        :return value:          (B) | (B x support_size)
        """
        hidden_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(hidden_state)
        if inv_transform:
            value = support_to_scalar(value_logits, self.support_limit)
            return policy_logits, hidden_state, value
        return policy_logits, hidden_state, value_logits

    def recurrent_inference(self,
                            hidden_state: torch.Tensor,
                            encoded_action: torch.Tensor,
                            scalar_transform: bool=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamics + Prediction function

        :param hidden_state:        (B x channels x h/16 x w/16) | (B x channels x h x w)
        :param encoded_action:      (B x 1 x h/16 x w/16) | (B x 1 x h x w)
        :param scalar_transform: whether to apply transformation on value and reward
        :return policy_logits:      (B x n_actions)
        :return next_hidden_state:  (B x channels x h/16 x w/16) | (B x channels x h x w)
        :return value:              (B) | (B x support_size)
        :return reward:             (B) | (B x support_size)
        """
        next_hidden_state, reward_logits = self.dynamics(hidden_state, encoded_action)
        policy_logits, value_logits = self.prediction(hidden_state)
        if scalar_transform:
            reward = support_to_scalar(reward_logits, self.support_limit)
            value = support_to_scalar(value_logits, self.support_limit)
            return policy_logits, next_hidden_state, value, reward
        return policy_logits, next_hidden_state, value_logits, reward_logits

    def set_weights(self, weights: Any) -> None:
        if weights is not None:
            self.load_state_dict(weights)


class MuZeroResidualNetwork(BaseNetwork):

    def __init__(self,
                 observation_dim: List[int],
                 n_actions: int,
                 n_stacked_observations: int,
                 blocks: int,
                 channels: int,
                 reduced_channels_reward: int,
                 reduced_channels_policy: int,
                 reduced_channels_value: int,
                 fc_reward_layers: List[int],
                 fc_policy_layers: List[int],
                 fc_value_layers: List[int],
                 downsample: bool,
                 support_limit: int) -> None:
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

        self.representation_network = RepresentationResidualNetwork(
            observation_dim, n_stacked_observations, blocks, channels, downsample
        )
        self.dynamics_network = DynamicsResidualNetwork(
            blocks,
            channels + 1,
            reduced_channels_reward,
            fc_reward_layers,
            support_size,
            block_output_size_reward
        )
        self.prediction_network = PredictionResidualNetwork(
            n_actions,
            blocks,
            channels,
            reduced_channels_policy,
            reduced_channels_value,
            fc_policy_layers,
            fc_value_layers,
            support_size,
            block_output_size_policy,
            block_output_size_value
        )


class MuZeroFullyConnectedNetwork(BaseNetwork):

    def __init__(self,
                 observation_dim: List[int],
                 n_actions: int,
                 n_stacked_observations: int,
                 encoding_size: int,
                 fc_reward_layers: List[int],
                 fc_policy_layers: List[int],
                 fc_value_layers: List[int],
                 fc_representation_layers: List[int],
                 fc_hidden_state_layers: List[int],
                 support_limit: int) -> None:
        super().__init__()
        self.support_limit = support_limit
        support_size = self.support_limit * 2 + 1

        self.representation_network = RepresentationFullyConnectedNetwork(
            observation_dim,
            n_stacked_observations,
            encoding_size,
            fc_representation_layers
        )
        self.dynamics_network = DynamicsFullyConnectedNetwork(
            n_actions, 
            encoding_size, 
            fc_reward_layers,
            fc_hidden_state_layers,
            support_size
        )
        self.prediction_network = PredictionFullyConnectedNetwork(
            n_actions,
            encoding_size,
            fc_policy_layers,
            fc_value_layers,
            support_size
        )


class DownSample(nn.Module):
    """Network to reduce the spatial resolution of observations before representation network"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        # (C_in x h x w) -> (C_out/2 x h/2 x w/2)
        self.conv1 = conv3x3(in_channels, int(out_channels / 2), stride=2)

        # (C_out/2 x h/2 x w/2) -> (C_out/2 x h/2 x w/2)
        self.res_block1 = nn.Sequential(*[
            ResidualBlock(int(out_channels / 2)) for _ in range(2)
        ])

        # (C_out/2 x h/2 x w/2) -> (C_out x h/4 x w/4)
        self.conv2 = conv3x3(int(out_channels / 2), out_channels, stride=2)

        # (C_out x h/4 x w/4) -> (C_out x h/4 x w/4)
        self.res_block2 = nn.Sequential(*[
            ResidualBlock(out_channels) for _ in range(3)
        ])

        # (C_out x h/4 x w/4) -> (C_out x h/8 x w/8)
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # (C_out x h/8 x w/8) -> (C_out x h/8 x w/8)
        self.res_block3 = nn.Sequential(*[
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


class RepresentationResidualNetwork(nn.Module):
    """Representation network

    If downsample is turned on, assuming that observation is in 3D (n_colors x h x w)
    and input is (stacked observations + encoded action):
        - input: (n_stacked_obs * n_colors + n_stacked_obs) x h x w
        - output: channels x downsampled_h x downsampled_w
    Otherwise, observation has shape (h * w) and input has shape (n_stacked_obs x h x w)

    :param observation_dim: Observation dimensionality
    :param n_stacked_observations: Number of observations stacked
    :param blocks: Number of residual blocks
    :param channels: Number of channels in the ResNet
    :param downsample: Whether to use downsample on inputted observation
    """

    def __init__(self,
                 observation_dim: List[int],
                 n_stacked_observations: int,
                 blocks: int,
                 channels: int,
                 downsample: bool) -> None:
        super().__init__()

        self.downsample = downsample
        if self.downsample:
            # C_in = n_stacked_obs * observation_dim[0] + n_stacked_obs
            # (C_in x h x w) -> (channels x h/16 x w/16)
            self.downsample_net = DownSample(
                n_stacked_observations * observation_dim[0],
                channels
            )
        else:
            # (n_stacked_obs x h x w) -> (channels x h x w)
            self.conv_block = ConvBlock(
                n_stacked_observations * observation_dim[0], channels)

        # preserves resolution
        self.res_tower = nn.Sequential(*[
            ResidualBlock(channels) for _ in range(blocks)
        ])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        :param observation:     (B, n_stacked_obs * obs_dim[0], h, w)
        :return hidden_state:   (B, channels, h/16, w/16) | (B, channels, h, w)
        """
        if self.downsample:
            out = self.downsample_net(observation)
        else:
            out = self.conv_block(observation)
        hidden_state = self.res_tower(out)
        return hidden_state


class DynamicsResidualNetwork(nn.Module):
    """Dynamics network"""

    def __init__(self,
                 blocks: int,
                 channels: int,
                 reduced_channels_reward: int,
                 fc_reward_layers: List[int],
                 support_size: int,
                 block_output_size_reward: int) -> None:
        super().__init__()
        self.conv_block = ConvBlock(channels, channels - 1)
        self.res_tower = nn.Sequential(*[
            ResidualBlock(channels - 1) for _ in range(blocks)
        ])

        self.reward_head = nn.Sequential(
            conv3x3(channels - 1, reduced_channels_reward, stride=1),
            nn.Flatten(start_dim=1),
            mlp([block_output_size_reward, *fc_reward_layers, support_size])
        )

    def forward(self, state_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param state_action:        (B, channels + 1, h/16, w/16) | (B, channels + 1, h, w)
        :return next_hidden_state:  (B, channels, h/16, w/16) | (B, channels, h, w)
        :return reward_logits:      (B, support_size)
        """
        out = self.conv_block(state_action)
        out = self.res_tower(out)
        next_hidden_state = out
        reward_logits = self.reward_head(out)
        return next_hidden_state, reward_logits


class PredictionResidualNetwork(nn.Module):
    """Prediction network"""

    def __init__(self,
                 n_actions: int,
                 blocks: int,
                 channels: int,
                 reduced_channels_policy: int,
                 reduced_channels_value: int,
                 fc_policy_layers: List[int],
                 fc_value_layers: List[int],
                 support_size: int,
                 block_output_size_policy: int,
                 block_output_size_value: int) -> None:
        super().__init__()
        self.res_tower = nn.Sequential(*[
            ResidualBlock(channels) for _ in range(blocks)
        ])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, reduced_channels_policy, 1),
            nn.Flatten(start_dim=1),
            mlp([block_output_size_policy, *fc_policy_layers, n_actions])
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, reduced_channels_value, 1),
            nn.Flatten(start_dim=1),
            mlp([block_output_size_value, *fc_value_layers, support_size])
        )

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param hidden_state:    (B, channels, h/16, w/16) | (B, channels, h, w)
        :return policy_logits:  (B, n_actions)
        :return value_logits:   (B, support_size)
        """
        out = self.res_tower(hidden_state)
        policy_logits = self.policy_head(out)
        value_logits = self.value_head(out)
        return policy_logits, value_logits


class RepresentationFullyConnectedNetwork(nn.Module):

    def __init__(self,
                 observation_dim: List[int],
                 n_stacked_observations: int,
                 encoding_size: int,
                 fc_representation_layers: List[int]) -> None:
        super().__init__()
        self.network = mlp([
            n_stacked_observations * math.prod(observation_dim),
            *fc_representation_layers,
            encoding_size
        ])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        :param observation:     (B, n_stacked_obs * obs_dim[0], obs_dim[1], obs_dim[2]))
        :return hidden_state:   (B, encoding_size)
        """
        hidden_state = self.network(observation.view(observation.shape[0], -1))
        return hidden_state


class DynamicsFullyConnectedNetwork(nn.Module):

    def __init__(self,
                 n_actions: int,
                 encoding_size: int,
                 fc_reward_layers: List[int],
                 fc_hidden_state_layers: List[int],
                 support_size: int) -> None:
        super().__init__()
        self.hidden_state_network = mlp([
            encoding_size + n_actions, *fc_hidden_state_layers, encoding_size
        ])
        self.reward_network = mlp([
            encoding_size + n_actions, *fc_reward_layers, support_size
        ])

    def forward(self, state_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param state_action:        (B, encoding_size + n_actions)
        :return next_hidden_state:  (B, encoding_size)
        :return reward_logits:      (B, support_size)
        """
        next_hidden_state = self.hidden_state_network(state_action)
        reward_logits = self.reward_network(state_action)
        return next_hidden_state, reward_logits


class PredictionFullyConnectedNetwork(nn.Module):

    def __init__(self,
                 n_actions: int,
                 encoding_size: int,
                 fc_policy_layers: List[int],
                 fc_value_layers: List[int],
                 support_size: int) -> None:
        super().__init__()
        self.policy_network = mlp([
            encoding_size, *fc_policy_layers, n_actions
        ])
        self.value_network = mlp([
            encoding_size, *fc_value_layers, support_size
        ])

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param hidden_state:    (B, encoding_size)
        :return policy_logits:  (B, n_actions)
        :return value_logits:   (B, support_size)
        """
        policy_logits = self.policy_network(hidden_state)
        value_logits = self.value_network(hidden_state)
        return policy_logits, value_logits
