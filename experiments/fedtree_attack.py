import os
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

from experiments.histogram_based_utils import fedtree_tree_dfs
from experiments.tree_based_utils import create_log, extract_categorical_features, extract_numerical_features
from xgboost_reconstruction import Attack


def fedtree_attack(log_path: str, args: dict, data: list, data_config: dict, force_attack: bool = False):
    defense = 'dp' in log_path
    create_log(log_path)

    with open(os.path.join(log_path, 'fedtree.log'), 'w') as f, redirect_stdout(f), redirect_stderr(f):
        fn, cat_fn, cat_values = extract_categorical_features(data, data_config)
        num_ranges = extract_numerical_features(data, data_config)

        x = pd.DataFrame(np.concatenate([data[i][0] for i in range(len(data))], axis=0), columns=fn)
        y = pd.DataFrame(np.concatenate([data[i][1] for i in range(len(data))], axis=0), columns=['target'])
        nClients = len(data)
        thresholds = [10000]

        experiment = Attack(
            data=x,
            target=y,
            fn=fn,
            log_path=log_path,
            thresholds=thresholds,
            nClients=nClients,
            categorical_features=cat_fn,
            categorical_values=cat_values,
            numerical_ranges=num_ranges,
            to_drop=args['to_drop'],
            tolerance=args['tolerance'],
            force=force_attack,
            baseline=args['baseline'],
        )

        base_path = "reconstructed_db"

        compromised_df = pd.DataFrame(data[args['compromised_cid']][0], columns=fn)

        if force_attack or not os.path.exists(os.path.join(log_path, 'reconstructed_db.pkl')):
            tree_dfs, xgb_info = fedtree_tree_dfs(args, data_config, defense=defense)
            print(f"Base score: {xgb_info.base_score}")
            print(f"Number of trees: {len(tree_dfs)}")
            print(f"Parameters: {xgb_info}")

            experiment.run_dfs(tree_dfs, xgb_info, name=base_path, compromised_df=compromised_df, evaluation_step=args['evaluation_step'])
        else:
            experiment.run_dfs(None, None, name=base_path, compromised_df=compromised_df, evaluation_step=args['evaluation_step'])