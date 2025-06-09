import json
import os
import re

import numpy as np
import pandas as pd

from xgboost_reconstruction.xgboost_utils import GenericXGBoostInfo


def fedtree_config(args, defense: bool = False):
    fedtree_directory: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fl_systems", "frameworks", "FedTree")
    data_directory = os.path.join(fedtree_directory, "..", "..", "..", "data", args['data'], f"nc_{args['num_clients']}", "data")
    config = json.load(open(os.path.join(data_directory, "config", "config.json")))

    custom_directory: str = os.path.join(fedtree_directory, "dataset", config["data"])  # dataset/__NAME__
    config_directory: str = os.path.join(fedtree_directory, "examples", config["data"])  # examples/__NAME__

    os.makedirs(custom_directory, exist_ok=True)
    os.makedirs(config_directory, exist_ok=True)

    __NAME__ = config["data"] + "_horizontal"
    __FEATURES__ = [config["data"]] + config["features"]
    __CAT_FEATURES__ = [config["data"]] + config["categorical_features"]

    subdirectories = [f.path for f in os.scandir(data_directory) if f.is_dir() and "client" in f.name]

    header = "id,y," + ",".join([f"x{i}" for i in range(1, len(__FEATURES__))])
    fmt = ','.join(['%d' if feature in __CAT_FEATURES__ else '%s' for feature in __FEATURES__])

    total_x_valid = []
    total_y_valid = []

    for idx, client_subdirectory in enumerate(subdirectories):
        x_train = np.load(os.path.join(client_subdirectory, "train", "x_train.npy"))
        y_train = np.load(os.path.join(client_subdirectory, "train", "y_train.npy"))
        x_valid = np.load(os.path.join(client_subdirectory, "valid", "x_valid.npy"))
        y_valid = np.load(os.path.join(client_subdirectory, "valid", "y_valid.npy"))

        # Convert categorical features and labels to int
        y_train = y_train.astype(int)
        y_valid = y_valid.astype(int)
        for i, feature in enumerate(__FEATURES__[1:]):
            if feature in __CAT_FEATURES__:
                x_train[:, i] = x_train[:, i].astype(int)
                x_valid[:, i] = x_valid[:, i].astype(int)

        total_x_valid.append(x_valid)
        total_y_valid.append(y_valid)

        np.savetxt(
            os.path.join(custom_directory, f"{__NAME__}_p{idx}.csv"),
            np.concatenate(
                (np.arange(len(y_train), dtype=np.int32).reshape(-1, 1), np.concatenate((y_train, x_train), axis=1)),
                axis=1),
            delimiter=",", header=header, comments="", fmt='%d,' + fmt
        )

        config_content = (f"data=./dataset/{config['data']}/{__NAME__}_p{idx}.csv\n"
                          f"test_data=./dataset/{config['data']}/{__NAME__}_test.csv\n"
                          f"n_parties={len(subdirectories)}\n"
                          f"num_class=2\n"
                          f"model_path=p{idx}.model\n"
                          f"objective=binary:logistic\n"
                          f"mode=horizontal\n"
                          f"data_format=csv\n"
                          f"privacy_tech={'none' if not defense else 'dp'}\n"
                          f"privacy_budget={args['differential_privacy']['epsilon'] if defense else 0}\n"
                          f"learning_rate={args['xgb_param']['eta']}\n"
                          f"max_depth={args['xgb_param']['max_depth']}\n"
                          f"n_trees={args['num_rounds']}\n"
                          f"lambda={args['xgb_param']['lambda']}\n"
                          f"gamma={args['xgb_param']['gamma']}\n"
                          f"ip_address=localhost\n")

        config_filename = os.path.join(config_directory, f"{__NAME__}_p{idx}.conf")
        with open(config_filename, 'w') as config_file:
            config_file.write(config_content)

    # Save validation data
    x_valid = np.concatenate(total_x_valid)
    y_valid = np.concatenate(total_y_valid)
    valid = np.hstack((y_valid, x_valid))

    np.savetxt(
        os.path.join(custom_directory, f"{__NAME__}_test.csv"),
        np.concatenate((np.arange(len(y_valid), dtype=np.int32).reshape(-1, 1), valid), axis=1),
        delimiter=",", header=header, comments="", fmt='%d,' + fmt
    )

    config_content = (f"test_data=./dataset/{config['data']}/{__NAME__}_test.csv\n"
                      f"n_parties={len(subdirectories)}\n"
                      f"num_class=2\n"
                      f"objective=binary:logistic\n"
                      f"mode=horizontal\n"
                      f"profiling=true\n"
                      f"verbose=1\n"
                      f"data_format=csv\n"
                      f"partition_mode=horizontal\n"
                      f"privacy_tech={'none' if not defense else 'dp'}\n"
                      f"privacy_budget={args['differential_privacy']['epsilon'] if defense else 0}\n"
                      f"learning_rate={args['xgb_param']['eta']}\n"
                      f"max_depth={args['xgb_param']['max_depth']}\n"
                      f"n_trees={args['num_rounds']}\n"
                      f"lambda={args['xgb_param']['lambda']}\n"
                      f"gamma={args['xgb_param']['gamma']}\n"
                      f"ip_address=localhost")

    config_filename = os.path.join(config_directory, f"{__NAME__}_server.conf")
    with open(config_filename, 'w') as config_file:
        config_file.write(config_content)


def fedtree_parse_source(file_path: str, feature_names: list[str], xgb_info: GenericXGBoostInfo, defense=False) -> dict:
    structure = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        stripped_line = line.strip()

        match = re.match(
            r"(\d+):\[f(\d+)<([\d.eE+-]+)] yes=(\d+),no=(\d+),missing=(\d+),sum_gh_pair=\(([\d.eE+-]+),([\d.eE+-]+)\),gain=([\d.eE+-]+)",
            stripped_line
        )

        leaf_match = re.match(
            r"(\d+):leaf=([\d.eE+-]+),sum_gh_pair=\(([\d.eE+-]+),([\d.eE+-]+)\),gain=([\d.eE+-]+)",
            stripped_line
        )

        if match:
            nid, fea_id, split_val, yes, no, missing, g_sum, h_sum, gain = match.groups()
            structure[int(nid)] = {
                'feature': feature_names[int(fea_id) - 1],
                'split': float(split_val),
                'yes': int(yes),
                'no': int(no),
                'missing': int(missing),
                'gain': float(gain),
                'g_sum': float(g_sum),
                'h_sum': float(h_sum) * (0.25 if defense else 1)
            }
        elif leaf_match:
            nid, leaf_val, g_sum, h_sum, gain = leaf_match.groups()
            structure[int(nid)] = {
                'feature': 'Leaf',
                'leaf_value': float(leaf_val) * xgb_info.lr,
                # IMPORTANT: Multiply by learning rate, because FedTree leaves are dumped already scaled by the learning rate
                'g_sum': float(g_sum),
                'h_sum': float(h_sum) * (0.25 if defense else 1)
            }

    return structure


def fedtree_create_dataframe(structure, xgb_info):
    data = []
    for nid, info in structure.items():
        if info.get('feature') == 'Leaf':
            yes_id, no_id, missing_id = None, None, None
            gain = info['leaf_value']
        else:
            yes_id = f"0-{info['yes']}"
            no_id = f"0-{info['no']}"
            missing_id = f"0-{info['missing']}"
            # gain = info['gain']
            gain = -1 * (info['g_sum'] / (info['h_sum'] + xgb_info.lambda_)) * xgb_info.lr

        cover = info['h_sum']

        data.append({
            'Tree': 0,
            'Node': nid,
            'ID': f"0-{nid}",
            'Feature': info.get('feature', 'Leaf'),
            'Split': info.get('split', None),
            'Yes': yes_id,
            'No': no_id,
            'Missing': missing_id,
            'Gain': gain,
            'Cover': cover,
            'Category': None
        })

    df = pd.DataFrame(data)
    df.sort_values(by=['Tree', 'Node'], inplace=True)
    return df


def fedtree_tree_dfs(args: dict, data_config: dict, defense=False) -> tuple[list[pd.DataFrame], GenericXGBoostInfo]:
    params = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fedtree_dir = os.path.join(current_dir, "..", "fl_systems", "frameworks", "FedTree", "client_messages")
    with open(os.path.join(fedtree_dir, "params_0.txt"), "r") as file:
        for line in file:
            if line.strip() == "":
                continue
            key, value = line.strip().split(": ")
            params[key] = value

    xgb_info = GenericXGBoostInfo(
        base_score=0.5,
        lambda_=float(params['lambda']),
        gamma=float(params['gamma']),
        lr=float(params['learning_rate'])
    )

    feature_names = data_config['features']

    tree_dfs = []
    for i in range(args['num_rounds']):
        structure = fedtree_parse_source(os.path.join(fedtree_dir, f"client_dump_0/tree_{i}_0.txt"), feature_names,
                                         xgb_info, defense=defense)
        df = fedtree_create_dataframe(structure, xgb_info)
        valid_ids = set(df['ID'])
        def is_leaf(row):
            return (row['Yes'] not in valid_ids) or (row['No'] not in valid_ids)
        df.loc[df.apply(is_leaf, axis=1), 'Feature'] = 'Leaf'
        tree_dfs.append(df)

    print(tree_dfs[0])

    return tree_dfs, xgb_info
