from player import Player, MuZeroPlayer, HumanPlayer
from game import Game, TicTacToe


class Arena:

    def __init__(self, p1: Player, p2: Player, game: Game) -> None:
        self.p1 = p1
        self.p2 = p2
        self.game = game


    def reset(self) -> None:
        self.game.reset()


    def _swapper(self):
        """Turn swapper"""
        while True:
            yield self.p1
            yield self.p2


    def run(self, verbose: bool=False) -> int:
        """Run a game and return the winner"""
        swapper = self._swapper()

        while True:
            player = next(swapper)
            if verbose:
                self.game.render()

            action = player.play(self.game)
            self.game.step(action)
            if self.game.terminated():
                if verbose:
                    self.game.render()
                return self.game.winner