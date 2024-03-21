import argparse

import game
from utils import VisitSoftmaxTemperatureFn
from muzero import MuZero


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuZero')
    parser.add_argument('--game-name', type=str, choices=['tictactoe'], default='tictactoe',
                        help='Environment name')
    tictactoe_args = parser.add_argument_group('TicTacToe arguments')
    tictactoe_args.add_argument('--tictactoe-size', type=int, default=3,
                                help='Board size')
    data_gen_args = parser.add_argument_group('Data generation arguments')
    data_gen_args.add_argument('--seed', type=int, default=0,
                                help='Seed for RNG')
    data_gen_args.add_argument('--games', type=int, default=200,
                                help='Number of self-play games to play')
    data_gen_args.add_argument('--gamma', type=float, default=1.0,
                                help='Discount factor')
    data_gen_args.add_argument('--max-moves', type=int, default=18,
                                help='Maximum total number of moves per game')
    data_gen_args.add_argument('--n-stacked-observations', type=int, default=1,
                                help='Number of previous observations (and previous actions) to add to the current observation')
    data_gen_args.add_argument('--buffer-size', type=int, default=10000,
                                help='Replay buffer size')
    data_gen_args.add_argument('--n-simulations', type=int, default=50,
                                help='Number of simulations to run MCTS')
    data_gen_args.add_argument('--dirichlet-alpha', type=float, default=1.1,
                                help='Alpha in Dirichlet(alpha) to sample noise, used for exploration (=10/max_pos_moves)')
    data_gen_args.add_argument('--exploration-frac', type=float, default=0.25,
                                help='Exploration fraction for Dirichlet noise injection')
    data_gen_args.add_argument('--c-base', type=int, default=19652,
                                help='PUCT parameter c_base')
    data_gen_args.add_argument('--c-init', type=float, default=1.25,
                                help='PUCT parameter c_init')
    network_training_args = parser.add_argument_group('Network training arguments')
    network_training_args.add_argument('--training-steps', type=int, default=1000,
                                        help='Number of steps to train the neural network')
    network_training_args.add_argument('--lr', type=float, default=1e-3,
                                        help='Learning rate for network optimizer')
    network_training_args.add_argument('--checkpoint-interval', type=int, default=10,
                                        help='')
    network_training_args.add_argument('--weight-decay', type=float, default=1e-4,
                                        help='L2 regularization parameter')
    network_training_args.add_argument('--batch-size', type=int, default=64,
                                        help='Mini-batch size')
    network_training_args.add_argument('--savedir', type=str,
                                        help='Directory to save the network')
    args = parser.parse_args()
    vst = VisitSoftmaxTemperatureFn()

    initializers = {
        'tictactoe': {
            'obj': game.TicTacToe(args.tictactoe_size),
            'players': 2,
            'action_space': list(range(args.tictactoe_size ** 2)),
            'visit_softmax_temperature_fn': vst.tictactoe,
        },
    }

    init = initializers[args.game_name]
    game = init['obj']
    args.players = init['players']
    args.action_space = init['action_space']
    args.visit_softmax_temperature_fn = init['visit_softmax_temperature_fn']
    del args.game_name

    agent = MuZero(game, args)
    agent.run()
