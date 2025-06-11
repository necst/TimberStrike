import json
import os
import pickle
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import xgboost as xgb

from experiments.tree_based_utils import extract_categorical_features, create_log, extract_numerical_features
from xgboost_reconstruction import XGBoostInfo, Attack


def fedxgbllr_attack(log_path: str, args: dict, data: list, data_config: dict,
                     force_attack: bool = False):
    create_log(log_path)
    victim_cid = args['victim_cid']

    with open(f"{args['path']}/aggregated_trees/{victim_cid}/model_0.pkl", 'rb') as f:
        classifiers = [cls for cls, _ in pickle.load(f)]

    cls: xgb.XGBClassifier = classifiers[victim_cid]
    num_trees = len(cls.get_booster().get_dump())
    base_score = XGBoostInfo(model=cls).base_score
    params = json.loads(cls.get_booster().save_config())['learner']['gradient_booster']['tree_train_param']

    with open(os.path.join(log_path, 'fedxgbllr.log'), 'w') as f, redirect_stdout(f), redirect_stderr(f):
        print(f"Base score: {base_score}")
        print(f"Number of trees: {num_trees}")
        print(f"Parameters: {params}")

        # set the feature names
        cls.get_booster().feature_names = data_config['features']

        _attack(log_path, args, cls, victim_cid, data, data_config, args['to_drop'], args['tolerance'], force_attack)


def _attack(log_path: str, args, victim_cls: xgb.XGBClassifier, victim_cid: int, data: list,
            data_config: dict, to_drop: list[str], tolerance: float, force_attack: bool):
    fn, cat_fn, cat_values = extract_categorical_features(data, data_config)
    num_ranges = extract_numerical_features(data, data_config)
    compromised_df = pd.DataFrame(data[args['compromised_cid']][0], columns=fn)
    data = data[victim_cid]
    x = pd.DataFrame(data[0], columns=fn)
    y = pd.DataFrame(data[1], columns=['target'])
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
        to_drop=to_drop,
        tolerance=tolerance,
        force=force_attack,
        baseline=args['baseline'],
    )

    base_path = "reconstructed_db"

    _, db = experiment.run(model=victim_cls, name=base_path, compromised_df=compromised_df, evaluation_step=args['evaluation_step'])
