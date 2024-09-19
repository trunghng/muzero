import argparse
import json
import os.path as osp
import sys
from types import SimpleNamespace

import torch

from games.game import Game
from games.tictactoe import TicTacToe
from games.cartpole import CartPole
from muzero import MuZero


def create_game(args) -> Game:
	if args.game == 'tictactoe':
		return TicTacToe(args.size)
	elif args.game == 'cartpole':
		if hasattr(args, 'render'):
			return CartPole(args.render)
		else:
			return CartPole()
	else:
		pass


def validate_args(parser, args):
	def validate1(required_tags):
		unspecified_tags = [k for k in required_tags if required_tags[k] is None]
		if unspecified_tags:
			parser.error(f'{", ".join(unspecified_tags)} must be specified.')

	def validate2(tag_value, value_wanted, tag, tag_dict):
		unspecified_tags = [k for k in tag_dict if tag_dict[k] is None]
		if tag_value == value_wanted and unspecified_tags:
			parser.error(f'{", ".join(unspecified_tags)} must be specified \
				when {tag}={value_wanted}.')

	validate1({
		'--workers': args.workers,
		'--seed': args.seed,
		'--max-moves': args.max_moves,
		'--simulations': args.simulations,
		'--gamma': args.gamma,
	})
	if args.mode == 'train':
		if args.game in ['tictactoe', 'cartpole'] and args.downsample:
			parser.error(f'Downsampling does not work with {args.game}.')
		validate1({
			'--n-stacked-observations': args.n_stacked_observations,
			'--network': args.network,
			'--batch-size': args.batch_size,
			'--checkpoint-interval': args.checkpoint_interval,
			'--buffer-size': args.buffer_size,
			'--td-steps': args.td_steps,
			'--unroll-steps': args.unroll_steps,
			'--training-steps': args.training_steps,
			'--optimizer': args.optimizer,
			'--lr': args.lr,
			'--weight-decay': args.weight_decay,
			'--lr-decay-rate': args.lr_decay_rate,
			'--lr-decay-steps': args.lr_decay_steps,
			'--support-limit': args.support_limit,
			'--value-loss-weight': args.value_loss_weight,
			'--reanalyse-workers': args.reanalyse_workers,
			'--target-network-update-freq': args.target_network_update_freq
		})
	else:
		validate1({'--tests': args.tests})
	validate2(args.game, 'tictactoe', '--game', {'--size': args.size})
	validate2(args.gumbel, True, '--gumbel', {
		'--max-considered-actions': args.max_considered_actions,
		'--c-visit': args.c_visit,
		'--c-scale': args.c_scale,
		'--gumbel-scale': args.gumbel_scale
	})
	validate2(args.gumbel, False, '--gumbel', {
		'--dirichlet-alpha': args.dirichlet_alpha,
		'--exploration-frac': args.exploration_frac,
		'--c-base': args.c_base,
		'--c-init': args.c_init
	})
	validate2(args.network, 'resnet', '--network', {
		'--blocks': args.blocks,
		'--channels': args.channels,
		'--reduced-channels-reward': args.reduced_channels_reward,
		'--reduced-channels-policy': args.reduced_channels_policy,
		'--reduced-channels-value': args.reduced_channels_value,
		'--resnet-fc-reward-layers': args.resnet_fc_reward_layers,
		'--resnet-fc-policy-layers': args.resnet_fc_policy_layers,
		'--resnet-fc-value-layers': args.resnet_fc_value_layers
	})
	validate2(args.network, 'resnet', '--mlp', {
		'--encoding-size': args.encoding_size,
		'--fc-reward-layers': args.fc_reward_layers,
		'--fc-policy-layers': args.fc_policy_layers,
		'--fc-value-layers': args.fc_value_layers,
		'--fc-representation-layers': args.fc_representation_layers,
		'--fc-hidden-state-layers': args.fc_hidden_state_layers
	})
	validate2(args.optimizer, 'SGD', '--optimizer', {
		'--momentum': args.momentum
	})


def main() -> None:
	parser = argparse.ArgumentParser(description='MuZero/Gumbel MuZero')

	mode_parsers = parser.add_subparsers(title='Modes')
	train_parser = mode_parsers.add_parser('train', formatter_class=argparse.RawTextHelpFormatter)
	train_parser.set_defaults(mode='train')
	test_parser = mode_parsers.add_parser('test', formatter_class=argparse.RawTextHelpFormatter)
	test_parser.set_defaults(mode='test')

	for p in [train_parser, test_parser]:
		p.add_argument('--game', type=str, choices=['tictactoe', 'cartpole'],
					   default='tictactoe', help='Game name')
		p.add_argument('--size', type=int,
					   help='Board size (if relevant)')
		p.add_argument('--exp-name', type=str, default='muzero',
					   help='Experiment name')
		p.add_argument('--workers', type=int,
					   help='Number of self-play workers')
		p.add_argument('--seed', type=int,
					   help='Seed for RNG')
		p.add_argument('--max-moves', type=int,
					   help='Maximum number of moves to end the game early')
		p.add_argument('--simulations', type=int,
					   help='Number of MCTS simulations')
		p.add_argument('--gamma', type=float,
					   help='Discount factor')
		p.add_argument('--gumbel', action='store_true',
					   help='')
		p.add_argument('--max-considered-actions', type=int,
					   help='Maximum number of actions sampled without replacement in Gumbel MuZero')
		p.add_argument('--c-visit', type=int,
					   help='')
		p.add_argument('--c-scale', type=float,
					   help='')
		p.add_argument('--gumbel-scale', type=float,
					   help='')
		p.add_argument('--dirichlet-alpha', type=float,
					   help='')
		p.add_argument('--exploration-frac', type=float,
					   help='')
		p.add_argument('--c-base', type=float,
					   help='')
		p.add_argument('--c-init', type=float,
					   help='')
		p.add_argument('--opponent', type=str, choices=['self', 'human', 'random'], default='random',
					   help='Opponent to test, or evalute in train mode:\n'
					   '   1. self: play with itself\n'
					   '   2. human: play with a human\n'
					   '   3. random: play with a random player')
		p.add_argument('--muzero-player', type=int, choices=[0, 1], default=0,
					   help="MuZero's turn order in test, or in evaluation during train:\n"
					   '   1. 0: plays first or MuZero is the only player (self-play or 1p games)\n'
					   '   2. 1: plays second')
		p.add_argument('--logdir', type=str,
					   help='Path to the log directory, which stores model file, config file, etc')
		p.add_argument('--config-path', type=str,
					   help='Path to the config file')

	train_parser.add_argument('--gpu', action='store_true',
							  help='Whether to enable GPU (if available)')
	train_parser.add_argument('--n-stacked-observations', type=int,
							  help='')
	train_parser.add_argument('--stack-action', action='store_true',
							  help='Whether to attach historical actions when stacking observations')
	train_parser.add_argument('--network', type=str, choices=['resnet', 'mlp'],
							  help='Network architecture of MuZero network')
	train_parser.add_argument('--blocks', type=int,
							  help='Number of residual blocks in the ResNet')
	train_parser.add_argument('--channels', type=int,
							  help='Number of channels in the ResNet')
	train_parser.add_argument('--reduced-channels-reward', type=int,
							  help='Number of channels in reward head')
	train_parser.add_argument('--reduced-channels-policy', type=int,
							  help='Number of channels in policy head')
	train_parser.add_argument('--reduced-channels-value', type=int,
							  help='Number of channels in value head')
	train_parser.add_argument('--resnet-fc-reward-layers', type=int, nargs='+',
							  help='Hidden layers in reward head')
	train_parser.add_argument('--resnet-fc-policy-layers', type=int, nargs='+',
							  help='Hidden layers in policy head')
	train_parser.add_argument('--resnet-fc-value-layers', type=int, nargs='+',
							  help='Hidden layers in value head')
	train_parser.add_argument('--encoding-size', type=int,
							  help='Observation encoding size')
	train_parser.add_argument('--fc-reward-layers', type=int, nargs='+',
							  help='')
	train_parser.add_argument('--fc-policy-layers', type=int, nargs='+',
							  help='')
	train_parser.add_argument('--fc-value-layers', type=int, nargs='+',
							  help='')
	train_parser.add_argument('--fc-representation-layers', type=int, nargs='+',
							  help='')
	train_parser.add_argument('--fc-hidden-state-layers', type=int, nargs='+',
							  help='')
	train_parser.add_argument('--downsample', action='store_true',
							  help='Whether to downsample observations before representation network')
	train_parser.add_argument('--batch-size', type=int,
							  help='Mini-batch size')
	train_parser.add_argument('--checkpoint-interval', type=int,
							  help='Checkpoint interval')
	train_parser.add_argument('--buffer-size', type=int,
							  help='Maximum number of self-play games to save in the replay buffer')
	train_parser.add_argument('--td-steps', type=int,
							  help='Number of steps in the future to take into account for '
							  'calculating the target value')
	train_parser.add_argument('--unroll-steps', type=int,
							  help='Number of unroll steps')
	train_parser.add_argument('--training-steps', type=int,
							  help='Number of training steps')
	train_parser.add_argument('--optimizer', type=str, choices=['SGD, Adam'],
							  help='Optimizer for network training')
	train_parser.add_argument('--lr', type=float,
							  help='Learning rate')
	train_parser.add_argument('--momentum', type=float,
							  help='Momentum factor, exclusively used for SGD optimizer')
	train_parser.add_argument('--weight-decay', type=float,
							  help='Weight decay')
	train_parser.add_argument('--lr-decay-rate', type=float,
							  help='Decay rate, used for exponential learning rate schedule')
	train_parser.add_argument('--lr-decay-steps', type=int,
							  help='Number of decay steps, used for exponential learning rate schedule')
	train_parser.add_argument('--support-limit', type=int,
							  help='Support limit')
	train_parser.add_argument('--value-loss-weight', type=float,
							  help='Weight of value loss in total loss function')
	train_parser.add_argument('--reanalyse-workers', type=int,
							  help='Number of reanalyse workers')
	train_parser.add_argument('--target-network-update-freq', type=int,
							  help='Target network update frequency, used in Reanalyse to provide a '
							  'fresher, stable target for the value function')
	train_parser.add_argument('--mcts-target-value', action='store_true',
							  help='Whether to use value function obtained from re-executing MCTS in '
							  'Reanalyse as target for training')

	test_parser.add_argument('--tests', type=int,
							 help='Number of games for testing')
	test_parser.add_argument('--render', action='store_true',
							 help='Whether to render each game during testing')
	args = parser.parse_args()

	if args.config_path is not None:
		with open(args.config_path) as f:
			args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
		if hasattr(args, 'comment'):
			del args.comment
	else:
		validate_args(parser, args)

	if args.mode == 'train':
		if hasattr(args, 'logdir') and args.logdir:
			try:
				with open(osp.join(args.logdir, 'config.json')) as f:
					config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
				config.logdir = args.logdir
				args = config
			except FileNotFoundError:
				print('Log directory not found')

		game = create_game(args)
		args.players = game.players
		args.observation_dim = game.observation_dim
		args.n_actions = game.n_actions
		args.visit_softmax_temperature_func = game.visit_softmax_temperature_func
		args.device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'
		# if args.network == 'resnet':
		# 	del args.encoding_size, args.fc_reward_layers, args.fc_policy_layers,\
		# 		args.fc_value_layers, args.fc_representation_layers, args.fc_hidden_state_layers
		# else:
		# 	del args.blocks, args.channels, args.reduced_channels_reward,\
		# 		args.reduced_channels_policy, args.reduced_channels_value,\
		# 		args.resnet_fc_reward_layers, args.resnet_fc_policy_layers,\
		# 		args.resnet_fc_value_layers
		muzero = MuZero(game, args)
		muzero.train()
	else:
		try:
			with open(osp.join(args.logdir, 'config.json')) as f:
				config = json.load(f)
		except TypeError:
			print('--logdir tag must be defined')
			sys.exit(0)
		except FileNotFoundError:
			print('Log directory not found')
			sys.exit(0)

		game = create_game(args)
		args.players = game.players
		args.observation_dim = game.observation_dim
		args.n_actions = game.n_actions
		args.n_stacked_observations = config['n_stacked_observations']
		args.network = config['network']
		if args.network == 'resnet':
			args.blocks = config['blocks']
			args.channels = config['channels']
			args.reduced_channels_reward = config['reduced_channels_reward']
			args.reduced_channels_policy = config['reduced_channels_policy']
			args.reduced_channels_value = config['reduced_channels_value']
			args.resnet_fc_reward_layers = config['resnet_fc_reward_layers']
			args.resnet_fc_policy_layers = config['resnet_fc_policy_layers']
			args.resnet_fc_value_layers = config['resnet_fc_value_layers']
			args.downsample = config['downsample']
		else:
			args.encoding_size = config['encoding_size']
			args.fc_reward_layers = config['fc_reward_layers']
			args.fc_policy_layers = config['fc_policy_layers']
			args.fc_value_layers = config['fc_value_layers']
			args.fc_representation_layers = config['fc_representation_layers']
			args.fc_hidden_state_layers = config['fc_hidden_state_layers']
		args.support_limit = config['support_limit']
		args.stack_action = config['stack_action']

		agent = MuZero(game, args)
		agent.test()


if __name__ == '__main__':
	main()
