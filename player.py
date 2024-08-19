from abc import ABC, abstractmethod
import random

from games.game import Game, BoardGame, ActType
from utils.game_utils import action_to_cellstr, cellstr_to_action


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
            if isinstance(game, BoardGame):
                actions = list(map(lambda a: ''.join(
                    action_to_cellstr(a, game.size)), game.legal_actions()))
                action_str = input(f'Legal moves are {actions}, choose one: ')
                action = cellstr_to_action(action_str, game.size)
            else:
                action_str = input(f'Legal moves are {actions}, choose one: ')
                action = int(action_str)
        return action
