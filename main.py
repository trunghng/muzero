import argparse

from game import Game, TicTacToe
from arena import Arena
from player import Player, RandomPlayer, HumanPlayer, MuZeroPlayer


def create_player(player: str) -> Player:
    if player == 'random':
        return RandomPlayer()
    elif player == 'human':
        return HumanPlayer()
    else:
        return MuZeroPlayer()


def create_game(name: str, boardsize: int) -> Game:
    if name == 'tictactoe':
        return TicTacToe(boardsize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuZero')

    mode_parsers = parser.add_subparsers(title='Modes')

    play_parser = mode_parsers.add_parser('play')
    play_parser.set_defaults(mode='play')
    player_choices = ['random', 'human', 'muzero']
    play_parser.add_argument('--p1', type=str, choices=player_choices, default='human',
                            help='Player 1')
    play_parser.add_argument('--p1-config', type=str, help='Config folder')
    play_parser.add_argument('--p2', type=str, choices=player_choices, default='human',
                            help='Player 2')

    train_parser = mode_parsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    train_parser.add_argument_group('Self-play arguments')
    train_parser.add_argument_group('Network training arguments')

    for p in [play_parser, train_parser]:
        p.add_argument('--game', type=str, default='tictactoe', help='Game name')
        p.add_argument('--boardsize', type=int, default=3, help='Board size (if relevant)')

    args = parser.parse_args()

    if args.mode == 'play':
        game = create_game(args.game, args.boardsize)

        if game.players == 2:
            p1 = create_player(args.p1)
            p2 = create_player(args.p2)
            arena = Arena(p1, p2, game)
            arena.run(True)
    elif args.mode == 'train':
        pass