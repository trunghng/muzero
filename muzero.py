import ray

from game import Game
from self_play import SelfPlay
from trainer import Trainer


class MuZero:
    
    def __init__(self, game: Game, config) -> None:
        ray.init()

        self.checkpoint = {
            'model_state_dict': None,
            'optimizer_state_dict': None,
            'episode_return': 0,
            'episode_length': 0,
            'lr': 0,
            'total_loss': 0,
            'value_loss': 0,
            'policy_loss': 0,
            'reward_loss': 0,
            'training_step': 0,
            'terminated': False
        }


    def train(self) -> None:
        pass



    
