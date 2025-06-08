#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <value_for_placeholder> <dataset_name>"
    exit 1
fi

if [ "$2" != "stroke" ] && [ "$2" != "diabetes" ]; then
    echo "Dataset name should be one of the following: stroke, diabetes"
    exit 1
fi

PLACEHOLDER_VALUE=$1
DATASET_NAME=$2

# shellcheck disable=SC2046
kill -9 $(lsof -ti:8080) 2> /dev/null

# No Defense Experiment
temp_file=$(mktemp)
# ./config_yaml/dataset_name/experiments_config.yaml
sed "s|\\\$MD_PLACEHOLDER\\\$|${PLACEHOLDER_VALUE}|g" ./config_yaml/"$DATASET_NAME"/experiments_config.yaml > "$temp_file"
python -m experiments.experiments "$temp_file"
rm "$temp_file"

# DP Experiment
others_values=(1 0.25 0.125)
ft_values=(200 50 25)

for (( i=0; i<${#others_values[@]}; i++ )); do
    others=${others_values[$i]}
    ft=${ft_values[$i]}
    echo "Processing and running config file: $temp_file, others: $others, ft: $ft"

    temp_file=$(mktemp)
    sed "s|\\\$MD_PLACEHOLDER\\\$|${PLACEHOLDER_VALUE}|g" ./config_yaml/"$DATASET_NAME"/experiments_config_dp.yaml > "$temp_file"

    temp_file_step2=$(mktemp)
    sed "s|\\\$DP_PLACEHOLDER\\\$|${others}|g" "$temp_file" > "$temp_file_step2"
    sed "s|\\\$FT_PLACEHOLDER\\\$|${ft}|g" "$temp_file_step2" > "$temp_file"

    python -m experiments.experiments "$temp_file"

    rm "$temp_file" "$temp_file_step2"
done