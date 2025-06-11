#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <num_clients> <max_depth> <dataset_name>"
    exit 1
fi

NUM_CLIENTS=$1
MAX_DEPTH=$2
DATASET_NAME=$3

# shellcheck disable=SC2046
kill -9 $(lsof -ti:8080) 2> /dev/null

# No Defense Experiment
temp_file=$(mktemp)
# ./config_yaml/dataset_name/experiments_config.yaml
sed "s|\\\$MD_PLACEHOLDER\\\$|${MAX_DEPTH}|g" ./config_yaml/"$DATASET_NAME"/experiments_config_"${NUM_CLIENTS}".yaml > "$temp_file"
python -m experiments.experiments "$temp_file"
rm "$temp_file"

# Baseline
temp_file=$(mktemp)
# ./config_yaml/dataset_name/experiments_config.yaml
sed "s|\\\$MD_PLACEHOLDER\\\$|${MAX_DEPTH}|g" ./config_yaml/"$DATASET_NAME"/experiments_config_"${NUM_CLIENTS}".yaml > "$temp_file"
sed -i 's/baseline: False/baseline: True/g' "$temp_file"
python -m experiments.experiments "$temp_file"
rm "$temp_file"

