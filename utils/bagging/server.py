import os
import warnings
from logging import INFO

import flwr as fl
from flwr.common.logger import log

from .fedxgb_bagging import FedXgbBagging
from .server_utils import (
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
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

    # Bagging training
    strategy = FedXgbBagging(
        evaluate_function=None,  # no need to centralised evaluation
        fraction_fit=(float(num_clients_per_round) / pool_size),
        min_fit_clients=num_clients_per_round,
        min_available_clients=pool_size,
        min_evaluate_clients=num_evaluate_clients if not centralised_eval else 0,
        fraction_evaluate=1.0 if not centralised_eval else 0.0,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=(
            evaluate_metrics_aggregation if not centralised_eval else None
        ),
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
    )

    metrics = history.metrics_distributed["F1"]

    best_round, best_f1 = max(metrics, key=lambda x: x[1])

    log(INFO, f"Best round: {best_round}, Best F1: {best_f1}")
