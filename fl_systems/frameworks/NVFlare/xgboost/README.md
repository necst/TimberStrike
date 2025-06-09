# Federated Learning for XGBoost

[Original Reference](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost)

## How to Build

See base [README](../../README.md) for general instructions on how to create the environment.

1) Install the required packages:

    ```bash
    # In this directory (and poetry shell active)
    pip install -r requirements.txt
    ```

2) Install the nvflare package:

    ```bash
    cd .. # In the root directory of NVFlare
    pip install -e . # -e: editable mode
    ```

## Run

   ```bash
   # In this directory
   ./prepare_and_run.sh
   ```
