from abc import ABC, abstractmethod
import random

from game import Game, ActType
from game_utils import cellstr_to_idx


class Player(ABC):
    """Player abstract class"""

    @abstractmethod
    def play(self, game: Game) -> ActType:
        """Select a move to play"""


class RandomPlayer(Player):
    """Player that plays moves randomly"""

    def play(self, game: Game) -> ActType:
        legal_actions = game.legal_actions()
        return random.choices(legal_actions)[0]


class HumanPlayer(Player):
    """Human player"""

    def play(self, game: Game) -> ActType:
        action = None
        while action not in game.legal_actions():
            actions = game.legal_actions()
            if game.type == 'board_game':
                actions = list(map(lambda a: ''.join(
                    idx_to_cellstr(a, game.size)), game.legal_actions()))
                action_str = input(f'Legal moves are {actions}, choose one: ')
                action = cellstr_to_idx(action_str, game.size)
            else:
                action_str = input(f'Legal moves are {actions}, choose one: ')
                action = int(action_str)
        return action
