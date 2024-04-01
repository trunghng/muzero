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
        action_str = input(f'Enter your move: ')
        return cellstr_to_idx(action_str, game.size)


class MuZeroPlayer(Player):
    """MuZero player"""

    def play(self, game: Game) -> ActType:
        pass