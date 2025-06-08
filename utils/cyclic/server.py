import os
import warnings
from logging import INFO

import flwr as fl
from flwr.common.logger import log

from .fedxgb_cyclic import FedXgbCyclic
from .server_utils import (
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    CyclicClientManager,
)
from .utils import server_args_parser

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    log(INFO, "Starting Flower server...")

    # Parse arguments for experimental settings
    args = server_args_parser()
    train_method = args.train_method
    pool_size = args.pool_size
    num_rounds = args.num_rounds
    num_clients_per_round = args.num_clients_per_round
    num_evaluate_clients = args.num_evaluate_clients
    centralised_eval = args.centralised_eval

    # Cyclic training
    strategy = FedXgbCyclic(
        fraction_fit=1.0,
        min_available_clients=pool_size,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
    )

    current_dir = os.path.dirname(__file__)
    cid_path = os.path.join(current_dir, "cids")
    if os.path.exists(cid_path):
        os.system(f"rm -rf {cid_path}")

    # Start Flower server
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_manager=CyclicClientManager() if train_method == "cyclic" else None,
    )

    metrics = history.metrics_distributed["F1"]

    best_round, best_f1 = max(metrics, key=lambda x: x[1])

    log(INFO, f"Best round: {best_round}, Best F1: {best_f1}")
