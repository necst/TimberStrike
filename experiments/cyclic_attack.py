import json
import os
import pickle
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import xgboost as xgb

from experiments.tree_based_utils import extract_categorical_features, extract_victim_boosters, \
    create_log, save_classes, _get_root_cover, evaluate_attack_with_noisy_data, extract_numerical_features
from xgboost_reconstruction import DatabaseReconstructor, run_attack_bagging_with_interleaving, evaluate_attack
from xgboost_reconstruction.xgboost_utils import GenericXGBoostInfo


def cyclic_attack(log_path: str, args: dict, data: list, data_config: dict,
                  force_attack: bool = False):
    create_log(log_path)

    victim_cid = args['victim_cid']
    with open(os.path.join(log_path, 'cyclic.log'), 'w') as f, redirect_stdout(f), redirect_stderr(f):
        fn, cat_fn, cat_values = extract_categorical_features(data, data_config)
        num_ranges = extract_numerical_features(data, data_config)

        compromised_df = pd.DataFrame(data[args['compromised_cid']][0], columns=fn)

        if force_attack or not os.path.exists(os.path.join(log_path, 'reconstructed_db.pkl')):
            with open(f"{args['path']}/global_model/round_{args['num_rounds'] * args['num_clients']}.pkl", 'rb') as f:
                global_model = pickle.load(f)

            model = xgb.Booster(model_file=global_model)

            base_score = json.loads(global_model)['learner']['learner_model_param']['base_score']
            params = args['xgb_param']
            xgb_info = GenericXGBoostInfo(base_score=base_score, lambda_=params['lambda'], lr=params['eta'],
                                          gamma=params['gamma'])

            print(f"XGBoost Info: {xgb_info}")
            print(f"Number of trees: {len(model.get_dump())}")

            victim_cls, trees, victim_cluster_class = extract_victim_boosters(log_path, model, args, victim_cid,
                                                                              params, _get_classes)

            print(f"Victim cluster class: {victim_cluster_class} <-> Victim cid: {victim_cid}")

            n_boosters = len(victim_cls.get_booster().get_dump())
            tree_dfs = [victim_cls.get_booster()[i].trees_to_dataframe() for i in range(n_boosters)]
            reconstructor = DatabaseReconstructor(tree_dfs, fn, xgb_info)

            print(f"Number of trees: {len(trees)}")
            print(f"Number of samples: {len(reconstructor.database.samples)}")
            print(f"Number of trees (reconstructor): {len(reconstructor.trees)}")

            log = os.path.join(log_path, 'log')
            os.makedirs(log, exist_ok=True)
            run_attack_bagging_with_interleaving(
                reconstructor,
                trees,
                victim_cluster_class,
                fn, params, log, data[victim_cid],
                fn, cat_fn, cat_values, num_ranges,
                tol=args['tolerance'], to_drop=args['to_drop'], compromised_df=compromised_df,
                cold_start=False,
                evaluation_step=args['evaluation_step'],
                baseline=args['baseline']
            )

            db = reconstructor.database

            os.makedirs(log_path, exist_ok=True)
            with open(os.path.join(log_path, 'reconstructed_db.pkl'), 'wb') as f:
                pickle.dump(db, f)
        else:
            with open(os.path.join(log_path, 'reconstructed_db.pkl'), 'rb') as f:
                db = pickle.load(f)

        print("Evaluating attack")
        evaluate_attack(log_path, db, data[victim_cid], fn, cat_fn, cat_values, num_ranges,
                        tol=args['tolerance'],
                        to_drop=args['to_drop'], compromised_df=compromised_df)


def _get_classes(log_path: str, model: xgb.Booster, args):
    """
    Cyclically assign trees to clients.
    """
    data = _get_root_cover(model)

    num_clients = args['num_clients']
    num_trees = len(model.get_dump())
    classes = [i % num_clients for i in range(num_trees)]

    tree_id = [i for i in range(num_trees)]
    data = [d[1] for d in data]

    data = np.column_stack((tree_id, data))

    save_classes(log_path, num_trees, data, classes)
    return classes
