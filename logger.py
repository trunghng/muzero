from datetime import datetime as dt
import json
import os
import os.path as osp
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch

from shared_storage import SharedStorage


class Logger:

    def __init__(self,
                 exp_name: str):
        self.log_dir = osp.join(os.getcwd(), 'data', exp_name)\
            if exp_name else f'/tmp/experiments/{str(dt.now())}'
        if not osp.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_file = open(osp.join(self.log_dir, 'loss.txt'), 'w')
        self.losses = []

    def save_config(self, config: Dict) -> None:
        del config['visit_softmax_temperature_func']
        output = json.dumps(config, separators=(',', ':\t'), indent=4)
        print('Experiment config:\n', output)
        with open(osp.join(self.log_dir, 'config.json'), 'w') as out:
            out.write(output)
        self.config = config

    def save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        torch.save(checkpoint, osp.join(self.log_dir, 'model.checkpoint'))

    def log(self, loss: float) -> None:
        self.losses.append(loss)
        self.log_file.write(str(loss) + '\n')
        self.log_file.flush()
        plt.plot(range(1, len(self.losses) + 1), self.losses, c='blue')
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.savefig(osp.join(self.log_dir, 'loss.png'))
        plt.close()
