from typing import Dict, Any

import ray

from game import Game, GameHistory
from mcts import MCTS
from network import MuZeroNetwork
from selfplay_utils import set_seed
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer


@ray.remote
class SelfPlay:

    def __init__(self,
                game: Game,
                initial_checkpoint: Dict[str, Any],
                config) -> None:
        set_seed(config.seed)
        self.config = config
        self.game = game
        self.mcts = MCTS(self.config)
        self.network = MuZeroNetwork(config)
        self.network.set_weights(initial_checkpoint['weights'])
        self.network.eval()


    def play_continuously(self,
                        shared_storage: SharedStorage,
                        replay_buffer: ReplayBuffer) -> None:
        while ray.get(shared_storage.get_info.remote('training_step') < self.config.training_steps) \
                and not ray.get(shared_storage.get_info.remote('terminated')):
            self.network.set_weights(ray.get(shared_storage.get_info.remote('model_state_dict')))

            game_history = self.play()
            replay_buffer.save_game.remote(game_history, shared_storage)


    def play(self) -> GameHistory:
            """Run a self-play game"""
            observation = self.game.reset()
            game_history = GameHistory()

            while True:
                stacked_observations = game_history.stack_observations(-1, 
                                        self.config.n_stacked_observations, len(self.config.action_space))
                root = self.mcts.search(self.network, stacked_observations, self.game.legal_actions(),
                                        game_history.actions, self.game.action_encoder, self.game.to_play)
                action = self.mcts.select_action(root, None)
                action_probs = self.mcts.action_probabilities(root)
                next_observation, reward, terminated = self.game.step(action)
                game_history.save(observation, action, reward, self.game.to_play, action_probs, root.value())
                observation = next_observation

                if terminated or len(game_history) > self.config.max_moves:
                    break
            return game_history