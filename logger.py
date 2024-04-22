import json
import os
from datetime import datetime as dt
from typing import Dict

from shared_storage import SharedStorage


class Logger:

    def __init__(self,
                log_dir: str,
                shared_storage: SharedStorage):
        self.log_dir = log_dir if log_dir else f'/tmp/experiments/{str(dt.now())}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.shared_storage_worker = shared_storage


    def save_config(self, config: Dict) -> None:
        del config['visit_softmax_temperature_func']
        output = json.dumps(config, separators=(',', ':\t'), indent=4)
        print('Experiment config:\n', output)
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as out:
            out.write(output)
        self.config = config
