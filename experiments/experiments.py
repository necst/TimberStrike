import argparse
import json
import os
import subprocess
import sys
import timeit

import numpy as np

from experiments.histogram_based_experiments import HistBasedPipeline
from experiments.params import experiments_config_parser
from experiments.tree_based_experiments import TreeBasedPipeline
from experiments.utils import link, PartitionArgs


def create_log(log_dir: str = 'results', train_dir: str = 'train', attack_dir: str = 'attack'):
    paths = [log_dir, os.path.join(log_dir, train_dir), os.path.join(log_dir, attack_dir)]
    for path in paths:
        os.makedirs(path, exist_ok=True)


def partition_data(log_file: str, partition_args: PartitionArgs, force_partition: bool = False):
    print(f"Partitioning data ({link(os.path.abspath(log_file))})")
    sys.argv = [
        'dataset_partitioner/data_generator.py', '-data', partition_args.data, '-niid_type', partition_args.niid_type,
        '-alpha',
        str(partition_args.alpha), '-num_clients', str(partition_args.num_clients), '-path', partition_args.path
    ]

    if os.path.exists(log_file) and not force_partition:
        return load_data(partition_args)
    with open(log_file, 'w') as f:
        # chmod +x dataset_partitioner/data_generator.py
        os.chmod('dataset_partitioner/data_generator.py', 0o755)
        subprocess.run(sys.argv, stdout=f, stderr=f)
        return load_data(partition_args)


def load_data(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', args.data, f'nc_{args.num_clients}', 'data')
    data = []
    for i in range(args.num_clients):
        x_train = np.load(os.path.join(data_path, f'client_{i}', 'train', 'x_train.npy'))
        y_train = np.load(os.path.join(data_path, f'client_{i}', 'train', 'y_train.npy'))
        x_test = np.load(os.path.join(data_path, f'client_{i}', 'valid', 'x_valid.npy'))
        y_test = np.load(os.path.join(data_path, f'client_{i}', 'valid', 'y_valid.npy'))
        data.append((x_train, y_train, x_test, y_test))
    config = json.load(open(os.path.join(data_path, 'config', 'config.json')))
    return data, config


def run_training_pipeline(defense=False):
    for name, train, attack in training_attack_pipeline:
        arg = args[name]
        if arg['skip']:
            continue
        start_time = timeit.default_timer()
        print(f"Training {name}")
        # if defense: append '_dp' to the log file name
        train_log = os.path.join(log_dir, 'train' + ('_dp' if defense else ''), f'{name}.log')
        print(f"Training {name} ({link(os.path.abspath(os.path.join(train_log)))})")
        ft = arg['force_training']
        train(train_log, arg, force_training=ft, defense=defense)
        print(f"{name} training time: {timeit.default_timer() - start_time} seconds")


def run_attack_pipeline(defense=False):
    for name, train, attack in training_attack_pipeline:
        arg = args[name]
        if arg['skip']:
            continue
        start_time = timeit.default_timer()
        print(f"Attacking {name}")
        attack_log = os.path.join(log_dir, 'attack' + ('_dp' if defense else ''), f'{name}')
        print(f"Attack {name} ({link(os.path.abspath(os.path.join(attack_log, f'{name}.log')))})")
        fa = arg['force_attack']
        attack(attack_log, arg, data, data_config, force_attack=fa)
        print(f"{name} attack time: {timeit.default_timer() - start_time} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('yaml_file', type=str, help='Path to the yaml file containing the experimental settings')
    args = parser.parse_args()
    return args.yaml_file


if __name__ == '__main__':
    # Parse arguments for experimental settings
    yaml_file_path = parse_args()
    args = experiments_config_parser(yaml_file_path)

    num_clients = args['global']['num_clients']
    force_partition = args['global']['force_partition']
    baseline = args['global']['baseline']
    niid_type = args['global']['niid_type']
    alpha = args['global']['alpha']
    data_path = args['global']['data_path']
    data = args['global']['data']
    defense = 'differential_privacy' in args['global']
    base_dir = f"{args['global']['log_dir']}/{data}"

    print("Starting the experiment with" + "out" * (not defense) + " defense")
    if defense:
        log_dir = os.path.join(base_dir, 'baseline' if baseline else '',
                               f"nc_{num_clients}", f"md_{str(args['global']['xgb_param']['max_depth'])}")
        os.makedirs(log_dir, exist_ok=True)
        log_dir = os.path.join(log_dir, f"dp_{str(args['global']['differential_privacy']['epsilon'])}")
        create_log(log_dir, train_dir='train_dp', attack_dir='attack_dp')
    else:
        log_dir = os.path.join(base_dir, 'baseline' if baseline else '',
                               f"nc_{num_clients}", f"md_{str(args['global']['xgb_param']['max_depth'])}")
        os.makedirs(log_dir, exist_ok=True)
        log_dir = os.path.join(log_dir, 'no_defense')
        create_log(log_dir)

    # Time the execution of tasks
    total_start_time = timeit.default_timer()

    # 1. Create and partition the dataset
    partition_args = PartitionArgs(niid_type=niid_type, alpha=alpha, num_clients=num_clients, data=data, path=data_path)
    start_time = timeit.default_timer()
    data, data_config = partition_data(os.path.join(base_dir, f"nc_{num_clients}", 'partition_data.log'), partition_args,
                                       force_partition=force_partition)
    print(f"Partitioning time: {timeit.default_timer() - start_time} seconds")

    training_attack_pipeline = TreeBasedPipeline(data=data, config=data_config).get_pipeline()
    training_attack_pipeline += HistBasedPipeline(data=data, config=data_config).get_pipeline()

    # 2.1 Training Pipeline
    run_training_pipeline(defense=defense)

    # 2.2 Attack Pipeline
    run_attack_pipeline(defense=defense)

    print(f"Total time to run: {timeit.default_timer() - total_start_time} seconds")
