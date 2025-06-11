import inspect
import json
import os
from functools import wraps

import matplotlib
import numpy as np
import pandas as pd
from diffprivlib import mechanisms

matplotlib.use('Agg')  # modify backend to use Agg to avoid tkinter error
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sympy import pprint

from experiments.tree_based_utils import PartitionType
from xgboost_reconstruction import Attack, Database, match_reconstruction


class PartitionArgs:
    def __init__(self, niid_type: PartitionType, alpha: float, num_clients: int, data: str, path: str):
        self.niid_type = niid_type
        self.alpha = alpha
        self.num_clients = num_clients
        self.data = data
        self.path = path


def skip_training_if_not_forced(func):
    @wraps(func)
    def wrapper(*args, force_training: bool = False, **kwargs):
        sig = inspect.signature(func)
        params = list(sig.parameters)
        is_method = params[0] in ('self', 'cls')
        log_file = args[1] if is_method else args[0]
        log_file = log_file.replace('/baseline/', '/')

        log_exists = os.path.exists(log_file)

        # Skip training if log exists and force_training is False
        if not force_training and log_exists:
            print(f"Skipping training as log file '{log_file}' exists and force_training is False.")
            return

        return func(*args, **kwargs)

    return wrapper


def link(path):
    # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
    return f"\033]8;;file://{path}\033\\{path}\033]8;;\033\\"
