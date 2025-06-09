#!/usr/bin/env bash
# change to "gpu_hist" for gpu training
TREE_METHOD="hist"

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <NUMBER_OF_CLIENTS> <ETA> <LAMBDA> <GAMMA> <MAX_DEPTH> <NUMBER_OF_ROUNDS>"
    exit 1
fi

n=$1
eta=$2
lambda=$3
gamma=$4
max_depth=$5
num_round=$6

prepare_job_config() {
    python3 utils/prepare_job_config.py --site_num "$1" --training_mode "$2" --split_method "$3" \
    --lr_mode "$4" --nthread 16 --tree_method "$5" --base "$6" --eta "$eta" --lambda "$lambda" --gamma "$gamma" --max_depth "$max_depth" --round_num "$num_round"
}

echo "Generating job configs"
prepare_job_config "$n" histogram_v2 uniform uniform $TREE_METHOD base_v2
echo "Job configs generated"
