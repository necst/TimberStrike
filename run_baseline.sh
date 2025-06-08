#!/bin/bash

# shellcheck disable=SC2046
kill -9 $(lsof -ti:8080) 2> /dev/null

DATASET_NAME=$1

max_depth_values=(3 4 5 6 7 8)

for md in "${max_depth_values[@]}"; do
    temp_file=$(mktemp)
    sed "s|\\\$MD_PLACEHOLDER\\\$|${md}|g" ./config_yaml/"$DATASET_NAME"/experiments_config.yaml > "$temp_file"
    sed -i 's/baseline: False/baseline: True/g' "$temp_file"
    python -m experiments.experiments "$temp_file"
    rm "$temp_file"

    # DP Experiment
    others_values=(0.001 0.005 2)
    ft_values=(25 50 200)

    for (( i=0; i<${#others_values[@]}; i++ )); do
        others=${others_values[$i]}
        ft=${ft_values[$i]}
        echo "Processing and running config file: $temp_file, others: $others, ft: $ft"

        temp_file=$(mktemp)
        sed "s|\\\$MD_PLACEHOLDER\\\$|${md}|g" ./config_yaml/"$DATASET_NAME"/experiments_config_dp.yaml > "$temp_file"
        sed -i 's/baseline: False/baseline: True/g' "$temp_file"

        temp_file_step2=$(mktemp)
        sed "s|\\\$DP_PLACEHOLDER\\\$|${others}|g" "$temp_file" > "$temp_file_step2"
        sed "s|\\\$FT_PLACEHOLDER\\\$|${ft}|g" "$temp_file_step2" > "$temp_file"

        python -m experiments.experiments "$temp_file"

        rm "$temp_file" "$temp_file_step2"
    done
done