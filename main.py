import argparse
import json
import os

import torch

from game import Game, TicTacToe
from muzero import MuZero


def create_game(args) -> Game:
    if args.game == 'tictactoe':
        return TicTacToe(args.size)
    elif args.game == 'cartpole':
        pass
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuZero')

    mode_parsers = parser.add_subparsers(title='Modes')
    train_parser = mode_parsers.add_parser('train')
    train_parser.set_defaults(mode='train')
    test_parser = mode_parsers.add_parser('test')
    test_parser.set_defaults(mode='test')

    for p in [train_parser, test_parser]:
        p.add_argument('--game', type=str, choices=['tictactoe', 'cartpole'],
                       default='tictactoe', help='Game name')
        p.add_argument('--size', type=int, default=3,
                       help='Board size (if relevant)')
        p.add_argument('--exp-name', type=str, default='muzero',
                       help='Experiment name')
        p.add_argument('--seed', type=int, default=0,
                       help='Seed for RNG')
        p.add_argument('--max-moves', type=int, default=9,
                       help='Maximum number of moves to end the game early')
        p.add_argument('--simulations', type=int, default=25,
                       help='Number of MCTS simulations')
        p.add_argument('--gamma', type=float, default=1,
                       help='Discount factor')
        p.add_argument('--root-dirichlet-alpha', type=float, default=0.1,
                       help='')
        p.add_argument('--root-exploration-fraction', type=float, default=0.25,
                       help='')
        p.add_argument('--c-base', type=float, default=19625,
                       help='')
        p.add_argument('--c-init', type=float, default=1.25,
                       help='')
        p.add_argument('--opponent', type=str, choices=['self', 'human', 'random'], default='self',
                       help='Opponent to test, or evalute in train mode:\n'
                       '   1. self: play with itself\n'
                       '   2. human: play with a human\n'
                       '   3. random: play with a random player')
        p.add_argument('--muzero-player', type=int, default=-1,
                       help="MuZero's turn order in test, or in evaluation during train:\n"
                       '    1. -1: plays first or MuZero is the only player (self-play or 1p games)\n'
                       '    2. 1: plays second')

    train_parser.add_argument('--gpu', action='store_true',
                              help='Whether to enable GPU (if available)')
    train_parser.add_argument('--workers', type=int, default=1,
                              help='Number of self-play workers')
    train_parser.add_argument('--stacked-observations', type=int, default=1,
                              help='')
    train_parser.add_argument('--stack-action', action='store_true',
                              help='Whether to attach historical actions when stacking observations')
    train_parser.add_argument('--blocks', type=int, default=1,
                              help='Number of residual blocks in the ResNet')
    train_parser.add_argument('--channels', type=int, default=16,
                              help='Number of channels in the ResNet')
    train_parser.add_argument('--reduced-channels-reward', type=int, default=16,
                              help='Number of channels in reward head')
    train_parser.add_argument('--reduced-channels-policy', type=int, default=16,
                              help='Number of channels in policy head')
    train_parser.add_argument('--reduced-channels-value', type=int, default=16,
                              help='Number of channels in value head')
    train_parser.add_argument('--fc-reward-layers', type=int, nargs='+', default=[8],
                              help='Hidden layers in reward head')
    train_parser.add_argument('--fc-policy-layers', type=int, nargs='+', default=[8],
                              help='Hidden layers in policy head')
    train_parser.add_argument('--fc-value-layers', type=int, nargs='+', default=[8],
                              help='Hidden layers in value head')
    train_parser.add_argument('--downsample', action='store_true',
                              help='Whether to downsample observations before representation network')
    train_parser.add_argument('--batch-size', type=int, default=256,
                              help='Mini-batch size')
    train_parser.add_argument('--checkpoint-interval', type=int, default=10,
                              help='Checkpoint interval')
    train_parser.add_argument('--buffer-size', type=int, default=3000,
                              help='Replay buffer size')
    train_parser.add_argument('--td-steps', type=int, default=9,
                              help='Number of steps in the future to take into account for calculating the target value')
    train_parser.add_argument('--unroll-steps', type=int, default=5,
                              help='Number of unroll steps')
    train_parser.add_argument('--training-steps', type=int, default=100000,
                              help='Number of training steps')
    train_parser.add_argument('--lr', type=float, default=0.003,
                              help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4,
                              help='Weight decay')
    train_parser.add_argument('--support-limit', type=int, default=10,
                              help='Support limit')
    train_parser.add_argument('--value-loss-weight', type=float, default=0.25,
                              help='Weight of value loss in total loss function')

    test_parser.add_argument('--n-tests', type=int, default=100,
                             help='Number of games for testing')
    test_parser.add_argument('--render', action='store_true',
                             help='Whether to render each game during testing')
    test_parser.add_argument('--log-dir', type=str,
                             help='Path to the log directory, which stores model file, config file, etc')

    args = parser.parse_args()

    game = create_game(args)
    args.players = game.players
    args.observation_dim = game.observation_dim
    args.action_space_size = game.action_space_size
    args.stack_action = game.stack_action
    args.visit_softmax_temperature_func = game.visit_softmax_temperature_func

    if args.mode == 'train':
        args.device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'
        muzero = MuZero(game, args)
        muzero.train()
    else:
        with open(os.path.join(args.log_dir, 'config.json')) as f:
            config = json.load(f)
        args.stacked_observations = config['stacked_observations']
        args.blocks = config['blocks']
        args.channels = config['channels']
        args.reduced_channels_reward = config['reduced_channels_reward']
        args.reduced_channels_policy = config['reduced_channels_policy']
        args.reduced_channels_value = config['reduced_channels_value']
        args.fc_reward_layers = config['fc_reward_layers']
        args.fc_policy_layers = config['fc_policy_layers']
        args.fc_value_layers = config['fc_value_layers']
        args.downsample = config['downsample']
        args.support_limit = config['support_limit']

        agent = MuZero(game, args)
        agent.test()
