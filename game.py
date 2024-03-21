from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar
import string

import numpy as np

import utils


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
PlayerType = TypeVar('PlayerType')


class Game(ABC):
    """Game abstract class"""

    def __init__(self, seed: int=None) -> None:
        """"""

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
    def to_play(self) -> PlayerType:
        """"""

    @abstractmethod
    def get_observation(self) -> ObsType:
        """"""

    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        """"""

    @abstractmethod
    def render(self) -> None:
        """"""


class GameHistory:
    """
    For non-board games, an action does not necessarily have a visible effect on 
    the observation, we encode historical actions into the stacked observation.
    """

    def __init__(self,
                game: Game,
                gamma: float,
                board_game: bool=True) -> None:
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.game = game
        self.gamma = gamma
        self.board_game = board_game
        
        self.store_experience(game.get_observation(), None, 0, game.to_play())


    def __len__(self) -> int:
        return len(self.action_history)


    def store_experience(self,
                        observation: ObsType,
                        action: ActType,
                        reward: float,
                        to_play: PlayerType) -> None:
        self.observation_history.append(observation)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.to_play_history.append(to_play)


    def store_search_statistics(self,
                                root,
                                action_space_size: int) -> None:
        pass


    def get_stacked_observations(self,
                                time_step: int,
                                n_stacked_observations: int,
                                action_space_size: int) -> np.ndarray:
        time_step = time_step % len(self)

        if self.board_game:

            def _stack_obs(to_play: PlayerType):
                obs_list = []
                for obs in reversed(self.observation_history[time_step - n_stacked_observations + 1: time_step + 1]):
                    obs_list.append(np.where(obs == to_play, 1, 0))

                return np.stack(obs_list)

            p1_planes = _stack_obs(-1)
            p2_planes = _stack_obs(1)
            to_play_plane = np.full(self.game.board.shape, self.to_play_history[time_step])

            return np.concatenate([p1_planes, p2_planes, np.expand_dims(to_play_plane, 0)])
        else:
            



class TicTacToe(Game):

    def __init__(self, size: int=3) -> None:
        self.size = size
        self.board = np.zeros((size, size))
        self.winner = None

        self._row_indices = string.ascii_lowercase
        self._col_indices = range(1, len(self._row_indices) + 1)
        self._pieces = {-1: 'X', 1: 'O', 0: ' '}


    def reset(self) -> ObsType:
        self.winner = None
        self.board = np.zeros((self.size, self.size))
        return self.get_observation()


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
        empty_cells = np.argwhere(state == 0)
        return [utils.cell2idx(c, self.size) for c in empty_cells]


    def to_play(self) -> PlayerType:
        nonzero = np.count_nonzero(self.board)
        return -1 if nonzero % 2 == 0 else 1


    def get_observation(self) -> ObsType:
        return self.board


    def step(self, action: ActType) -> Tuple[ObsType, float, bool]:
        self.board[utils.idx2cell(action, self.size)] = self.to_play()
        terminated = self.terminated()
        reward = 1 if self.winner is not None else 0
        return self.get_observation(), reward, terminated
    

    def render(self) -> None:
        col_indices_txt = '  '
        rows, cols = self.size, self.size
        for i in range(cols):
            col_indices_txt += '  ' + str(self._col_indices[i]) + ' '

        print(col_indices_txt)
        print('  +' + '---+' * cols)
        for i in range(rows):
            board_row = self._row_indices[i] + ' | '
            for j in range(cols):
                piece = self._pieces[self.board[i, j]]
                board_row += piece + ' | '
            print(board_row)
            print('  +' + '---+' * cols)

if __name__ == '__main__':
    g = TicTacToe()
    h = GameHistory(g, 0)
    g.render()
    a = 0
    o, r, d = g.step(a)
    t = g.to_play()
    h.store_experience(o, a, r, t)
    g.render()
    a = 5
    o, r, d = g.step(a)
    t = g.to_play()
    h.store_experience(o, a, r, t)
    g.render()
    a = 3
    o, r, d = g.step(a)
    t = g.to_play()
    h.store_experience(o, a, r, t)
    g.render()
    a = 6
    o, r, d = g.step(a)
    t = g.to_play()
    h.store_experience(o, a, r, t)
    g.render()
    print(h.get_stacked_observations(3, 2, 9))

