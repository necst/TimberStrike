#!/usr/bin/env bash
OUTPUT_PATH="/tmp/nvflare/xgboost_dataset"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ -z "$1" ] || [ -z "$2" ]
then
    echo "Usage: $0 <NUMBER_OF_CLIENTS> <DATASET_PATH>"
    exit 1
fi

n=$1
dataset_path=$2


echo "Generating data splits for ${n} clients"

if [ ! -f "${dataset_path}" ]
then
    echo "Please check if you saved stroke dataset in ${dataset_path}"
fi

mkdir -p ${OUTPUT_PATH}
mkdir -p ${OUTPUT_PATH}/"${n}"_clients

echo "Generated stroke data splits, reading from ${dataset_path}"

python3 "${SCRIPT_DIR}"/utils/prepare_data_split.py \
--data_path "${dataset_path}" \
--site_num "${n}" \
--out_path "${OUTPUT_PATH}/${n}_clients" \
#        --size_total 7775 \
#        --size_valid 600 \
#        --split_method ${split_mode} \

echo "Data splits are generated in ${OUTPUT_PATH}"
