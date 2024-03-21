from abc import ABC, abstractmethod
import random

from game import Game, ActType

class Player(ABC):
    """Player abstract class"""

    @abstractmethod
    def play(self, game: Game) -> ActType:
        """Select a move to play"""