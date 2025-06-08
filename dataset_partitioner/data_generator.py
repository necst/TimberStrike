#!/usr/bin/env python

import argparse
import json
import os
import warnings

import numpy as np

from dataset_partitioner.data_loader import load_stroke, load_diabetes
from dataset_partitioner.data_partitioner import split_iid, split_label, split_sample, split_feature
from dataset_partitioner.dataset_utils import get_config

warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument('-data', default='stroke', help='Type of data',
                    type=str)
parser.add_argument("-niid_type", choices=['iid', 'label', 'sample', 'feature'], default='iid',
                    help="Heterogeneity type", type=str)
parser.add_argument('-alpha', default=0.5, help=" alpha for non-iid (sigma for noise)",
                    type=float)  # small alpha for non-IID
parser.add_argument('-num_clients', default=5, help="Number of clients", type=int)
parser.add_argument('-path', default='./healthcare-dataset-stroke-prep.csv', help="path to the dataset", type=str)
args = parser.parse_args()

data = args.data
niid_type = args.niid_type
alpha = args.alpha
path = args.path
num_clients = args.num_clients


def main():
    if data == 'stroke':
        print('Loading Stroke')
        x_train, y_train, x_valid, y_valid = load_stroke(path)
    elif data == 'diabetes':
        print('Loading Diabetes')
        x_train, y_train, x_valid, y_valid = load_diabetes(path)
    else:
        print("No other dataset implemented yet")
        exit()

    # split the validation dataset
    split_iid(x_valid, y_valid, args, num_clients, type='valid')

    # split the training dataset
    if niid_type == 'iid':
        print('Splitting IID')
        split_iid(x_train, y_train, args, num_clients, type='train')
    elif niid_type == 'label':
        print('Splitting Label imbalance')
        split_label(x_train, y_train, args, num_clients)
    elif niid_type == 'sample':
        print('Splitting Sample imbalance')
        split_sample(x_train, y_train, args, num_clients)
    elif niid_type == 'feature':
        print('Splitting Feature imbalance')
        split_feature(x_train, y_train, args, num_clients)

    # save data info to json for PS
    n_classes = np.unique(y_valid, axis=0).shape[0] if np.unique(y_valid, axis=0).shape[0] > 2 else 1
    data_info = {
        'input_shape': x_train.shape[1:],
        'num_classes': n_classes,  # np.unique(y_valid, axis=0).shape[0],
        'data': args.data,
        'niid_type': args.niid_type,
        'alpha': args.alpha
    }
    _dir = f"data/{data}/nc_{num_clients}/data/server/"
    os.makedirs(_dir, exist_ok=True)
    with open(_dir + "data_info.json", "w") as outfile:
        json.dump(data_info, outfile)

    # Save a config file
    _dir = f"data/{data}/nc_{num_clients}/data/config/"
    os.makedirs(_dir, exist_ok=True)
    with open(_dir + "config.json", "w") as outfile:
        json.dump(get_config(data, path), outfile)


if __name__ == '__main__':
    main()
