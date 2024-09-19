from typing import List, Tuple

import numpy as np

from games.game import BoardGame, ObsType
from utils.game_utils import action_to_cell, cell_to_action


class TicTacToe(BoardGame):

    def __init__(self, size: int=3) -> None:
        super().__init__(size, [3, size, size])

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

    def legal_actions(self) -> np.ndarray:
        empty_cells = np.argwhere(self.board == 0)
        empty_cells = [cell_to_action(c, self.size) for c in empty_cells]
        actions = np.zeros(self.size ** 2)
        actions[empty_cells] = 1
        return actions

    def step(self, action: int) -> Tuple[ObsType, float, bool]:
        self.board[action_to_cell(action, self.size)] = self._to_play
        self._to_play *= -1
        terminated = self.terminated()
        reward = 1 if (self.winner is not None and self.winner != 0) else 0
        return self.observation(), reward, terminated

    def observation(self) -> ObsType:
        p1_plane = np.where(self.board == -1, 1, 0)
        p2_plane = np.where(self.board == 1, 1, 0)
        to_play_plane = np.full_like(self.board, self._to_play)
        return np.array([p1_plane, p2_plane, to_play_plane])

    def action_encoder(self, action: int) -> int:
        one_hot_action = np.zeros((self.size, self.size))
        one_hot_action[action_to_cell(action, self.size)] = 1
        return one_hot_action

    def visit_softmax_temperature_func(self,
                                       training_steps: int,
                                       training_step: int) -> float:
        """
        :param training_steps: number of training steps
        :param training_step: current training step
        """
        return 1.0

    def to_play(self) -> int:
        return 0 if self._to_play == -1 else 1
