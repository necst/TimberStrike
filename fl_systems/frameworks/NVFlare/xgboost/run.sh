#!/bin/bash
set -e

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <NUMBER_OF_CLIENTS> <ETA> <LAMBDA> <GAMMA> <MAX_DEPTH> <NUMBER_OF_ROUNDS> <DATASET_PATH>"
    exit 1
fi

n=${1:-2}
eta=${2:-0.3}
lambda=${3:-1}
gamma=${4:-0}
max_depth=${5:-7}
num_round=${6:-100}
dataset_path=$7

## Prepare data
echo "Preparing data"
bash prepare_data.sh "${n}" "${dataset_path}"

# Prepare job config
echo "Preparing job config"
bash prepare_job_config.sh "${n}" "${eta}" "${lambda}" "${gamma}" "${max_depth}" "${num_round}"

# Run XGBoost
echo "Running XGBoost"
study=histogram_v2_uniform_split_uniform_lr
nvflare simulator jobs/"${n}"_${study} -w "${PWD}"/workspaces/xgboost_workspace_"${n}"_${study} -n "${n}" -t "${n}"
