import json
import os
import warnings
from logging import INFO

import flwr as fl
import numpy as np
import pandas as pd
import xgboost as xgb
from flwr.common.logger import log

from .client_utils import XgbClient
from .utils import client_args_parser, BST_PARAMS, NUM_LOCAL_ROUND, DATA_PATH

warnings.filterwarnings("ignore", category=UserWarning)


def delete_saved_models():
    os.system("rm -rf cyclic/first_round_model")
    os.system("rm -rf cyclic/global_model")


if __name__ == "__main__":
    log(INFO, "Starting XGBoost client...")

    # Delete saved models
    delete_saved_models()

    # Parse arguments for experimental settings
    args = client_args_parser()

    # Train method (bagging or cyclic)
    train_method = args.train_method

    log(INFO, f"Training method: {train_method}")
    log(INFO, f"Number of local rounds: {NUM_LOCAL_ROUND}")

    base_path = DATA_PATH

    log(INFO, f"Loading dataset from {base_path}...")

    x_train = np.load(os.path.join(base_path, f"client_{args.partition_id}", "train", "x_train.npy"))
    y_train = np.load(os.path.join(base_path, f"client_{args.partition_id}", "train", "y_train.npy"))
    x_val = np.load(os.path.join(base_path, f"client_{args.partition_id}", "valid", "x_valid.npy"))
    y_val = np.load(os.path.join(base_path, f"client_{args.partition_id}", "valid", "y_valid.npy"))

    config = json.load(open(os.path.join(base_path, "config", "config.json")))

    fn = config["features"]
    label = config["label"]

    df_train = pd.DataFrame(x_train, columns=fn)
    df_val = pd.DataFrame(x_val, columns=fn)
    df_train_labels = pd.DataFrame(y_train, columns=[label])
    df_val_labels = pd.DataFrame(y_val, columns=[label])

    # Create DMatrix
    train_dmatrix = xgb.DMatrix(df_train, label=df_train_labels)
    valid_dmatrix = xgb.DMatrix(df_val, label=df_val_labels)

    num_train = len(y_train)
    num_val = len(y_val)

    log(INFO, "Dataset loaded.")
    log(INFO, f"Shape of training data: {x_train.shape}")
    log(INFO, f"Shape of training labels: {y_train.shape}")
    log(INFO, f"Shape of validation data: {x_val.shape}")
    log(INFO, f"Shape of validation labels: {y_val.shape}")

    # Hyper-parameters for xgboost training
    num_local_round = NUM_LOCAL_ROUND
    params = BST_PARAMS

    # Setup learning rate
    if args.train_method == "bagging" and args.scaled_lr:
        new_lr = params["eta"] / args.num_partitions
        params.update({"eta": new_lr})

    # Start Flower client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=XgbClient(
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            num_local_round,
            params,
            train_method,
            node_id=args.partition_id,
        ),
    )
