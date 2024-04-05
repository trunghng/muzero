import argparse

from game import Game, TicTacToe
from arena import Arena
from player import Player, RandomPlayer, HumanPlayer, MuZeroPlayer
from muzero import MuZero


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
    selfplay_args = train_parser.add_argument_group('Self-play arguments')
    selfplay_args.add_argument('--seed', type=int, default=0,
                                help='Seed')
    selfplay_args.add_argument('--workers', type=int, default=1,
                                help='Number of self-play workers')
    selfplay_args.add_argument('--max-moves', type=int, default=9,
                                help='Maximum number of moves to end the game early')
    selfplay_args.add_argument('--stacked-observations', type=int, default=1,
                                help='')
    selfplay_args.add_argument('--stack-action', action='store_true',
                                help='Whether to attach historical actions when stacking observations')
    selfplay_args.add_argument('--simulations', type=int, default=25,
                                help='Number of MCTS simulations')
    selfplay_args.add_argument('--gamma', type=float, default=1,
                                help='Discount factor')
    selfplay_args.add_argument('--root-dirichlet-alpha', type=float, default=0.1,
                                help='')
    selfplay_args.add_argument('--root-exploration-fraction', type=float, default=0.25,
                                help='')
    selfplay_args.add_argument('--c-base', type=float, default=19625,
                                help='')
    selfplay_args.add_argument('--c-init', type=float, default=1.25,
                                help='')

    network_args = train_parser.add_argument_group('Network training arguments')
    network_args.add_argument('--blocks', type=int, default=1,
                                help='Number of residual blocks in the ResNet')
    network_args.add_argument('--channels', type=int, default=16,
                                help='Number of channels in the ResNet')
    network_args.add_argument('--reduced-channels-reward', type=int, default=16,
                                help='Number of channels in reward head')
    network_args.add_argument('--reduced-channels-policy', type=int, default=16,
                                help='Number of channels in policy head')
    network_args.add_argument('--reduced-channels-value', type=int, default=16,
                                help='Number of channels in value head')
    network_args.add_argument('--fc-reward-layers', type=int, nargs='+', default=[8],
                                help='Hidden layers in reward head')
    network_args.add_argument('--fc-policy-layers', type=int, nargs='+', default=[8],
                                help='Hidden layers in policy head')
    network_args.add_argument('--fc-value-layers', type=int, nargs='+', default=[8],
                                help='Hidden layers in value head')
    network_args.add_argument('--downsample', action='store_true',
                                help='Whether to downsample observations before representation network')
    network_args.add_argument('--batch-size', type=int, default=64,
                                help='Mini-batch size')
    network_args.add_argument('--buffer-size', type=int, default=3000,
                                help='Replay buffer size')
    network_args.add_argument('--td-steps', type=int, default=20,
                                help='Number of steps in the future to take into account for calculating the target value')
    network_args.add_argument('--unroll-steps', type=int,
                                help='Number of unroll steps')
    network_args.add_argument('--training-steps', type=int, default=1000000,
                                help='Number of training steps')
    network_args.add_argument('--lr', type=float, default=0.003,
                                help='Learning rate')
    network_args.add_argument('--weight-decay', type=float, default=1e-4,
                                help='Weight decay')
    network_args.add_argument('--support-limit', type=int, default=10,
                                help='Support limit')
    network_args.add_argument('--value-loss-weight', type=float, default=0.25,
                                help='Weight of value loss in total loss function')
    network_args.add_argument('--save-model', action='store_true',
                                help='Whether to save the model')

    for p in [play_parser, train_parser]:
        p.add_argument('--game', type=str, default='tictactoe', help='Game name')
        p.add_argument('--boardsize', type=int, default=3, help='Board size (if relevant)')

    args = parser.parse_args()

    game = create_game(args.game, args.boardsize)
    args.players = game.players
    args.observation_dim = game.observation_dim
    args.action_space = game.action_space
    args.visit_softmax_temperature_func = game.visit_softmax_temperature_func

    if args.mode == 'play':
        if game.players == 2:
            p1 = create_player(args.p1)
            p2 = create_player(args.p2)
            arena = Arena(p1, p2, game)
            arena.run(True)
    elif args.mode == 'train':
        muzero = MuZero(game, args)
        muzero.train()
        