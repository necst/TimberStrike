import json
import os
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

# modify backend to use Agg to avoid tkinter error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from xgboost_reconstruction import RecoverXGBClassifier, DatabaseReconstructor, database_to_final_dataframe, \
    match_reconstruction, Database
from xgboost_reconstruction.run_experiment import get_result, tsne_plot, evaluate_attack


class PartitionType:
    IID = 'iid'
    FEATURE = 'feature'
    LABEL = 'label'
    SAMPLE = 'sample'


class ClassesCidMapper:
    def __init__(self, num_clients: int, classes: list, cids: list):
        classes = classes[:num_clients]
        cids = cids[:num_clients]
        self.cid_to_class = {cid: cls for cid, cls in zip(cids, classes)}
        self.class_to_cid = {cls: cid for cls, cid in zip(classes, cids)}

    def get_cid(self, cls: int) -> int:
        return self.class_to_cid[cls]

    def get_class(self, cid: int) -> int:
        return self.cid_to_class[cid]


def create_log(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

def evaluate_attack_with_noisy_data(log_path: str, db: Database, args:dict, fn: list, cat_fn: list, cat_values: dict, num_ranges: dict, tol: float = 0.1, to_drop: list = None, compromised_df = None):
    if 'differential_privacy' in args:
        epsilon = args['differential_privacy']['epsilon']
        log_path = os.path.join(log_path, 'dp')
        os.makedirs(log_path, exist_ok=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dp_dir = os.path.join(current_dir, '..', f'data_dp_{epsilon}')

        x = np.load(os.path.join(dp_dir, f"client_{args['victim_cid']}", 'train', 'x_train.npy'))
        y = np.load(os.path.join(dp_dir, f"client_{args['victim_cid']}", 'train', 'y_train.npy'))

        compromised_df = np.load(os.path.join(dp_dir, f"client_{args['compromised_cid']}", 'train', 'x_train.npy'))
        compromised_df = pd.DataFrame(compromised_df, columns=fn)

        # call evaluate_attack
        data = [x, y]

        evaluate_attack(log_path, db, data, fn, cat_fn, cat_values, num_ranges, tol=tol, to_drop=to_drop, compromised_df=compromised_df)

def extract_categorical_features(data: list, data_config: dict):
    fn = data_config['features']
    num_fn = data_config['numerical_features']

    cat_fn = [f for f in fn if f not in num_fn]
    cat_values = {name: np.unique(np.concatenate([client[0][:, fn.index(name)] for client in data])) for name in cat_fn}

    for name in cat_values.keys():
        cat_values[name] = np.sort(cat_values[name])

    return fn, cat_fn, cat_values

def extract_numerical_features(data: list, data_config: dict) -> dict:
    fn = data_config['features']
    num_fn = data_config['numerical_features']

    # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

    x = np.concatenate([client[0] for client in data])
    num_ranges = {name: [x[:, fn.index(name)].min(), x[:, fn.index(name)].max()] for name in num_fn}

    return num_ranges


def extract_victim_boosters(log_path: str, model: xgb.Booster, args, victim_cid: int,
                            params: dict, get_classes: callable
                            ) -> tuple[XGBClassifier, list[dict[str, bytearray | Any]], int]:
    classes = get_classes(log_path, model, args)
    cids, mapper = get_real_cids(args["path"], args["num_clients"], classes)

    print(f"Real cids: {cids}")

    victim_cluster_class = mapper.get_class(victim_cid)
    victim_boosters = []

    for booster, c in zip(model, classes):
        if c == victim_cluster_class:
            victim_boosters.append(booster)

    cls = RecoverXGBClassifier(victim_boosters).recover_xgb()

    cls.get_booster().set_param(params=params)

    trees = []
    for booster, c in zip(model, classes):
        trees.append({"cid": c, "tree": booster.save_raw('json')})

    return cls, trees, victim_cluster_class


def save_classes(log_path: str, num_trees: int, data: list, classes: list):
    fig, ax = plt.subplots()
    ax.scatter(range(num_trees), [d[1] for d in data], c=classes)
    for i, txt in enumerate(classes):
        txt = f'{txt}, {i}'
        ax.annotate(txt, (i, data[i][1]))
    plt.xlabel('Tree ID')
    plt.ylabel('Cover of Root Node')
    plt.title('Classes of Trees')
    plt.savefig(f'{log_path}/classes.png')


def get_real_cids(base_path: str, num_clients: int, rec_classes: list) -> tuple[list, ClassesCidMapper]:
    with open(f'{base_path}/cids/cid_list.pkl', 'rb') as f:
        cids = pickle.load(f)
    mapper = ClassesCidMapper(num_clients, rec_classes, cids)

    return cids, mapper


def fedxgbllr_config(args: dict, defense: bool = False) -> tuple[str, str]:
    num_clients = args['num_clients']
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if defense:
        data_dir = os.path.join(current_dir, '..', 'data', args['data'], f'data_dp_{args["differential_privacy"]["epsilon"]}')
    else:
        data_dir = os.path.join(current_dir, '..', 'data', args['data'], f'nc_{args["num_clients"]}', 'data')

    config = json.load(open(os.path.join(data_dir, 'config', 'config.json')))
    dataset = config['data']

    fexgbllr_dir = os.path.join(current_dir, '..', 'utils', 'fedxgbllr')
    conf_dir = os.path.join(fexgbllr_dir, 'conf')

    # create a yaml file in conf/dataset called dataset.yaml
    with open(os.path.join(conf_dir, 'dataset', f'{dataset}.yaml'), 'w') as f:
        yaml.dump({'defaults': [{'task': 'Binary_Classification'}],
                   'dataset_name': dataset,
                   'early_stop_patience_rounds': 10,
                   'data_path': os.path.abspath(data_dir),
                   'already_partitioned': True}, f)

    # create a yaml file in conf/clients called dataset_num_clients_clients.yaml
    os.makedirs(os.path.join(conf_dir, 'clients'), exist_ok=True)
    with open(os.path.join(conf_dir, 'clients', f'{dataset}_{num_clients}_clients.yaml'), 'w') as f:
        yaml.dump({'n_estimators_client': args['num_rounds'],
                   'num_rounds': 30,
                   'client_num': num_clients,
                   'num_iterations': 500,
                   'xgb': args['xgb_param'],
                   'CNN': {'lr': .0005}}, f)

    return dataset, f'{dataset}_{num_clients}_clients'


def bagging_cyclic_config(dir: str, args: dict, defense: bool = False):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    dir = os.path.join(current_dir, '..', 'utils', dir)

    xgb_param = {
        "objective": "binary:logistic",
        "eta": float(args['xgb_param']['eta']),
        "max_depth": args['xgb_param']['max_depth'],
        "gamma": args['xgb_param']['gamma'],
        "lambda": args['xgb_param']['lambda'],
        "eval_metric": "auc",
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 1,
        "tree_method": "hist",
    }
    if defense:
        data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', args['data'], f'data_dp_{args["differential_privacy"]["epsilon"]}'))
    else:
        data_path = os.path.abspath(os.path.join(current_dir, '..', 'data', args['data'], f'nc_{args["num_clients"]}', 'data'))

    # params will be xgb_param and a string "data_path" that will be absolute path to data folder or data_dp folder
    params = {
        "xgb_param": xgb_param,
        "data_path": data_path
    }

    with open(os.path.join(dir, 'params.json'), 'w') as f:
        json.dump(params, f)


def _get_root_cover(model: xgb.Booster):
    """
    Get the cover of the root node for each tree.
    """
    num_trees = len(model.get_dump())
    data = []

    for tid in range(num_trees):
        dump = model.get_dump(with_stats=True)[tid]
        regex = re.compile(r'cover=(\d+\.\d+)')
        cover = regex.search(dump).group(1)
        regex_leaf = re.compile(r'leaf')
        num_leaf = len(regex_leaf.findall(dump))
        data.append((tid, float(cover), num_leaf))

    return data
