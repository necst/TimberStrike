import os
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import xgboost as xgb

from experiments.tree_based_utils import create_log, extract_categorical_features, extract_numerical_features
from xgboost_reconstruction import Experiment


def nvflare_attack(log_path: str, args: dict, data: list, data_config: dict, force_attack: bool = False):
    create_log(log_path)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    nvflare_dir = os.path.join(current_dir, '..', 'fl_systems', 'frameworks', 'NVFlare', 'xgboost', 'workspaces')

    name = f"xgboost_workspace_{args['num_clients']}"
    workspace = [f.path for f in os.scandir(nvflare_dir) if f.is_dir() and name in f.name][0]
    model_path = os.path.join(workspace, 'site-1', 'simulate_job', 'model.json')

    with open(os.path.join(log_path, 'nvflare.log'), 'w') as f, redirect_stdout(f), redirect_stderr(f):
        to_drop = args['to_drop']
        tolerance = args['tolerance']

        fn, cat_fn, cat_values = extract_categorical_features(data, data_config)
        num_ranges = extract_numerical_features(data, data_config)

        x = pd.DataFrame(np.concatenate([data[i][0] for i in range(len(data))], axis=0), columns=fn)
        y = pd.DataFrame(np.concatenate([data[i][1] for i in range(len(data))], axis=0), columns=['target'])
        nClients = len(data)
        thresholds = [10000]

        print(f"x.shape: {x.shape}, y.shape: {y.shape}")

        experiment = Experiment(
            data=x,
            target=y,
            fn=fn,
            log_path=log_path,
            thresholds=thresholds,
            nClients=nClients,
            categorical_features=cat_fn,
            categorical_values=cat_values,
            numerical_ranges=num_ranges,
            to_drop=to_drop,
            tolerance=tolerance,
            force=force_attack,
            baseline=args['baseline'],
        )

        base_path = "reconstructed_db"

        compromised_df = pd.DataFrame(data[args['compromised_cid']][0], columns=fn)

        if force_attack or not os.path.exists(os.path.join(log_path, 'reconstructed_db.pkl')):
            model: xgb.XGBClassifier = xgb.XGBClassifier()
            model.load_model(model_path)

            print(f"Number of trees: {len(model.get_booster().get_dump())}")

            print(f"Base score: {model.base_score}")
            print(model.get_booster().feature_names)
            _, db = experiment.run(model=model, name=base_path, compromised_df=compromised_df, evaluation_step=args['evaluation_step'])
        else:
            _, db = experiment.run(model=None, name=base_path, compromised_df=compromised_df, evaluation_step=args['evaluation_step'])
