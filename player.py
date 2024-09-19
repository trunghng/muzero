from abc import ABC, abstractmethod

import random

from games.game import Game, BoardGame
from utils.game_utils import action_to_cellstr, cellstr_to_action, mask_illegal_actions


class Player(ABC):
    """Player abstract class"""

    @abstractmethod
    def play(self, game: Game) -> int:
        """Select a move to play"""


class RandomPlayer(Player):
    """Player that plays moves randomly"""

    def play(self, game: Game) -> int:
        legal_actions = mask_illegal_actions(game.legal_actions())
        return random.choice(legal_actions)


class HumanPlayer(Player):
    """Human player"""

    def play(self, game: Game) -> int:
        action = None
        actions = mask_illegal_actions(game.legal_actions())
        actions_display = list(map(lambda a: ''.join(
            action_to_cellstr(a, game.size)), actions)
        ) if isinstance(game, BoardGame) else actions

        while action not in actions:
            action_str = input(f'Legal moves are {actions_display}, choose one: ')
            if isinstance(game, BoardGame):
                action = cellstr_to_action(action_str, game.size)
            else:
                action = int(action_str)
        return action
