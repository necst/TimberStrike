import importlib
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Tuple, Dict

import numpy as np

from experiments.bagging_attack import bagging_attack
from experiments.cyclic_attack import cyclic_attack
from experiments.fedxgbllr_attack import fedxgbllr_attack
from experiments.tree_based_utils import fedxgbllr_config, bagging_cyclic_config
from experiments.utils import skip_training_if_not_forced


class TreeBasedPipeline:
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                 config: Dict):
        self.data = data
        self.config = config

    def get_pipeline(self):
        return [
            ('cyclic', self.train_cyclic, cyclic_attack),
            ('bagging', self.train_bagging, bagging_attack),
            ('fedxgbllr', self.train_fedxgbllr, fedxgbllr_attack)
        ]

    @skip_training_if_not_forced
    def train_fedxgbllr(self, log_file: str, args, defense=False):
        original_cwd = os.getcwd()

        dataset, clients = fedxgbllr_config(args, defense=defense)

        try:
            os.chdir('fl_systems/utils')
            sys.argv = ['fl_systems/utils/fedxgbllr/main.py', f'dataset={dataset}', f'clients={clients}']
            sys.path.append(os.getcwd())

            log_file = os.path.join(original_cwd, log_file)

            with open(log_file, 'w') as f, redirect_stdout(f), redirect_stderr(f):
                mod = importlib.import_module('fedxgbllr.main')
                if hasattr(mod, 'main'):
                    mod.main()
                    time.sleep(5)
        finally:
            os.chdir(original_cwd)
            sys.path.remove(os.getcwd())

        print("FedXGBLLR training completed and output logged.")

    @skip_training_if_not_forced
    def train_bagging(self, log_file: str, args, defense=False):
        command = ['fl_systems/utils/bagging/run.sh', str(args["num_clients"]), str(args["num_rounds"])]

        bagging_cyclic_config('bagging', args, defense)  # PASS DATA_DP or DATA based on defense, use abs path

        with open(log_file, 'w') as f:
            subprocess.run(command, stdout=f, stderr=f)

        print("Bagging training completed and output logged.")

    @skip_training_if_not_forced
    def train_cyclic(self, log_file: str, args, defense=False):
        command = ['fl_systems/utils/cyclic/run.sh', str(args["num_clients"]), str(args["num_rounds"] * args["num_clients"])] # num_rounds * num_clients = total rounds, so that each client gets to train num_rounds trees

        bagging_cyclic_config('cyclic', args, defense)  # PASS DATA_DP or DATA based on defense, use abs path

        with open(log_file, 'w') as f:
            subprocess.run(command, stdout=f, stderr=f)

        print("Cyclic training completed and output logged.")
