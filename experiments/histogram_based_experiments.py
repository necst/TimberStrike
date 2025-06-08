import os
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np

from experiments.fedtree_attack import fedtree_attack
from experiments.histogram_based_utils import fedtree_config
from experiments.nvflare_attack import nvflare_attack
from experiments.utils import skip_training_if_not_forced


class HistBasedPipeline:
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                 config: Dict):
        self.data = data
        self.config = config

    def get_pipeline(self):
        return [
            ('fedtree', self.train_fedtree, fedtree_attack),
            ('nvflare', self.train_nvflare, nvflare_attack)
        ]

    @skip_training_if_not_forced
    def train_fedtree(self, log_file: str, args, defense: bool = False):
        cwd = os.path.dirname(os.path.abspath(__file__))

        fedtree_config(args, defense=defense)

        sys.argv = [
            os.path.join(cwd, '..', 'frameworks', 'FedTree', 'run.sh'), str(args["num_clients"]), str(args["data"])
        ]

        with open(log_file, 'w') as f:
            os.chmod(os.path.join(cwd, '..', 'frameworks', 'FedTree', 'run.sh'), 0o755)
            subprocess.run(sys.argv, stdout=f, stderr=f)

    @skip_training_if_not_forced
    def train_nvflare(self, log_file: str, args, defense: bool = False):
        cwd = os.path.dirname(os.path.abspath(__file__))

        eta = args["xgb_param"]["eta"]
        gamma = args["xgb_param"]["gamma"]
        lambda_ = args["xgb_param"]["lambda"]
        max_depth = args["xgb_param"]["max_depth"]
        num_rounds = args["num_rounds"]
        if defense:
            dataset_path = os.path.abspath(os.path.join(cwd, '..', 'data', args['data'], f'data_dp_{args["differential_privacy"]["epsilon"]}'))
        else:
            dataset_path = os.path.abspath(os.path.join(cwd, '..', 'data', args["data"], f'nc_{args["num_clients"]}', 'data'))

        sys.argv = [
            os.path.join(cwd, '..', 'frameworks', 'NVFlare', 'xgboost', 'run.sh'), str(args["num_clients"]), str(eta), str(gamma),
            str(lambda_), str(max_depth), str(num_rounds), dataset_path
        ]

        with open(log_file, 'w') as f:
            os.chmod(os.path.join(cwd, '..', 'frameworks', 'NVFlare', 'xgboost', 'run.sh'), 0o755)
            subprocess.run(sys.argv, stdout=f, stderr=f)
