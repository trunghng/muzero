from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar

import numpy as np

from game_utils import idx_to_cell, cell_to_idx, draw_board


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
PlayerType = TypeVar('PlayerType')


class Game(ABC):
    """Game abstract class"""

    def __init__(self,
                 players: int,
                 observation_dim: List[int],
                 action_space_size: int,
                 stack_action: bool,
                 game_type: str) -> None:
        self.players = players
        self.observation_dim = observation_dim
        self.action_space_size = action_space_size
        self.stack_action = stack_action
        self.type = game_type

    @abstractmethod
    def reset(self) -> ObsType:
        """"""

    @abstractmethod
    def terminated(self) -> bool:
        """"""

    @abstractmethod
    def legal_actions(self) -> List[ActType]:
        """"""

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        """"""

    @abstractmethod
    def observation(self) -> ObsType:
        """"""

    @abstractmethod
    def action_encoder(self, action: ActType) -> ActType:
        """"""

    @abstractmethod
    def visit_softmax_temperature_func(self,
                                       training_steps: int,
                                       training_step: int) -> float:
        """
        :param training_steps: number of training steps
        :param training_step: current training step
        """

    @abstractmethod
    def render(self) -> None:
        """"""


class GameHistory:
    """
    For atari games, an action does not necessarily have a visible effect on
    the observation, we encode historical actions into the stacked observation.
    """

    def __init__(self, game: Game) -> None:
        self.observations = []              # o_t: State observations
        self.actions = []                   # a_{t+1}: Action leading from s_t -> s_{t+1}
        self.encoded_actions = []
        self.rewards = []                   # u_{t+1}: Observed reward after performing a_{t+1}
        self.to_plays = []                  # p_t: Player to play
        self.action_probabilities = []      # pi_t: Action probabilities produced by MCTS
        self.root_values = []               # v_t: MCTS value estimation
        self.reanalysed_root_values = []
        self.action_encoder = game.action_encoder
        self.initial_observation = game.observation()

    def __len__(self) -> int:
        return len(self.observations)

    def save(self,
             observation: ObsType,
             action: ActType,
             reward: float,
             to_play: PlayerType,
             pi: List[float],
             root_value: float) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.encoded_actions.append(self.action_encoder(action))
        self.rewards.append(reward)
        self.to_plays.append(to_play)
        self.action_probabilities.append(pi)
        self.root_values.append(root_value)

    def stack_n_observations(self,
                             t: int,
                             n: int,
                             action_space_size: int,
                             stack_action: bool) -> np.ndarray:
        """
        Stack n most recent observations (and corresponding actions lead
        to the states with Atari) upto 't': o_{t-n+1}, ..., o_t

        :param t: time step of the latest observation to stack
        :param n: number of observations to stack
        :param action_space_size: size of the action space
        :param stack_action: whether to stack historical actions
        """
        planes = []
        if len(self) == 0:
            planes.append(self.initial_observation)
            if stack_action:
                planes.append(np.zeros_like(self.initial_observation))
            for _ in range(n - 1):
                planes.append(np.zeros_like(self.initial_observation))
                if stack_action:
                    planes.append(np.zeros_like(self.initial_observation))
        else:
            # Convert to positive index
            t = t % len(self)
            n_ = min(n, t + 1)

            for step in reversed(range(t - n_ + 1, t + 1)):
                planes.append(self.observations[step])
                if stack_action:
                    planes.append(np.full_like(self.observations[step],
                        self.actions[step] / action_space_size))

            # If n_stack_observations > t + 1, we attach planes of zeros instead
            for _ in range(n - n_):
                planes.append(np.zeros_like(self.observations[step]))
                if stack_action:
                    planes.append(np.zeros_like(self.observations[step]))

        return np.concatenate(planes, axis=0)

    def compute_return(self, gamma: float, player: PlayerType, players: int) -> float:
        """
        Compute episode return w.r.t the perspective of the player,
        assuming that the game is over
            G = r_1 + gamma * r_2 + ... + gamma^{T-1} * r_T

        :param gamma: discount factor
        :param player: player turn
        :param players: number of players
        """
        def __get_reward(reward, time_step):
            if players == 2 and ((time_step % 2 == 0 and player == 1)\
            or (time_step % 2 == 1 and player == -1)):
                return -reward
            else:
                return reward

        eps_return = __get_reward(self.rewards[-1], len(self.observations) - 1)
        for i, r in enumerate(reversed(self.rewards[0:-1])):
            reward = __get_reward(r, len(self.observations) - i)
            eps_return = eps_return * gamma + reward
        return eps_return

    def make_target(
            self,
            t: int,
            td_steps: int,
            gamma: float,
            unroll_steps: int,
            action_space_size: int
        ) -> Tuple[List[float], List[float], List[List[float]]]:
        """
        Create targets for every unroll steps

        :param t: current time step
        :param td_steps: n-step TD
        :param gamma: discount factor
        :param unroll_steps: number of unroll steps
        :param action_space_size: size of the action space
        :return: value targets, reward targets, policy targets
        """
        value_targets, reward_targets, policy_targets = [], [], []

        def _compute_value_target(step: int) -> float:
            """
            Compute value target
            - For board games, value target is the total reward from the current index til the end
            - For other games, value target is the discounted root value of the search tree
            'td_steps' into the future, plus the discounted sum of all rewards until then

            z_t = u_{t+1} + gamma * u_{t+2} + ... + gamma^{n-1} * u_{t+n} + gamma^n * v_{t+n}
            """
            if gamma == 1:
                rewards = []
                for i in range(step, len(self)):
                    rewards.append(self.rewards[i] if self.to_plays[i] == self.to_plays[step] else -self.rewards[i])
                value = sum(rewards)
            else:
                bootstrap_step = step + td_steps
                if bootstrap_step < len(self):
                    bootstrap = self.root_values[bootstrap_step] if self.to_plays[bootstrap_step] == self.to_plays[step]\
                                    else -self.root_values[bootstrap_step]
                    bootstrap *= gamma ** td_steps
                else:
                    bootstrap = 0

                discounted_rewards = [
                    (self.rewards[k] if self.to_plays[step + k] == self.to_plays[step] else -reward) * gamma ** k
                    for k in range(step + 1, bootstrap_step + 1)
                ]
                value = sum(discounted_rewards) + bootstrap
            return value

        for step in range(t, t + unroll_steps + 1):
            value = _compute_value_target(step)

            if step < len(self):
                value_targets.append(value)
                reward_targets.append(self.rewards[step])
                policy_targets.append(self.action_probabilities[step])
            else:
                value_targets.append(0)
                reward_targets.append(0)
                policy_targets.append([1 / action_space_size] * action_space_size)

        return value_targets, reward_targets, policy_targets


class TicTacToe(Game):

    def __init__(self, size: int=3) -> None:
        super().__init__(players=2,
                         observation_dim=[3, 3, 3],
                         action_space_size=size**2,
                         stack_action=False,
                         game_type='board_game')
        self.size = size
        # -1, 1, 0 denote X, O, empty respectively
        self.board = np.zeros((size, size))
        # X moves first
        self.to_play = -1
        self.winner = None

    def reset(self) -> ObsType:
        self.board = np.zeros((self.size, self.size))
        self.to_play = -1
        self.winner = None
        return self.observation()

    def terminated(self) -> bool:
        """Whether the game is terminated"""
        sums = []
        terminated = False

        col_sums = np.sum(self.board, axis=0)
        for s in col_sums:
            sums.append(s)

        row_sums = np.sum(self.board, axis=1)
        for s in row_sums:
            sums.append(s)

        diag_sum = np.trace(self.board)
        sums.append(diag_sum)

        anti_diag_sum = np.trace(np.fliplr(self.board))
        sums.append(anti_diag_sum)

        for s in sums:
            if abs(s) == self.size:
                terminated = True
                self.winner = int(s / self.size)
                break

        # tie checking
        if np.sum(np.abs(self.board)) == self.board.size:
            terminated = True
            self.winner = 0

        return terminated

    def legal_actions(self) -> List[ActType]:
        empty_cells = np.argwhere(self.board == 0)
        return [cell_to_idx(c, self.size) for c in empty_cells]

    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        self.board[idx_to_cell(action, self.size)] = self.to_play
        self.to_play *= -1
        terminated = self.terminated()
        reward = 1 if (self.winner is not None and self.winner != 0) else 0
        return self.observation(), reward, terminated

    def observation(self) -> ObsType:
        p1_plane = np.where(self.board == -1, 1, 0)
        p2_plane = np.where(self.board == 1, 1, 0)
        to_play_plane = np.full_like(self.board, self.to_play)
        return np.array([p1_plane, p2_plane, to_play_plane])

    def action_encoder(self, action: ActType) -> ActType:
        one_hot_action = np.zeros((self.size, self.size))
        one_hot_action[idx_to_cell(action, self.size)] = 1
        return one_hot_action

    def visit_softmax_temperature_func(self,
                                       training_steps: int,
                                       training_step: int) -> float:
        """
        :param training_steps: number of training steps
        :param training_step: current training step
        """
        return 1.0

    def render(self) -> None:
        draw_board(self.board, {-1: 'X', 1: 'O', 0: ' '})
