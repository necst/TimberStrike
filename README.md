# TimberStrike

This repository contains the code for the experiments in the paper **"TimberStrike: Dataset Reconstruction Attack Revealing Privacy Leakage in Federated Tree-Based Systems"**.

---

## 📁 Project Structure

```bash
.
├── config_yaml/               # Configuration files for experiments
├── dataset_partitioner/       # Logic for simulating client-side data partitioning
├── Dockerfile                 # Docker container configuration
├── experiments/               # Training and attack workflow implementations
├── fl_systems/                # Federated learning systems integration
│   ├── frameworks/            # Included frameworks: FedTree (v1.0.5, latest) and NVFlare (v2.5)
│   └── utils/                 # Federated XGBoost via Flower (i.e., bagging, cyclic, FedXGBllr)
├── paper_visualization/       # Scripts for generating plots and figures used in the paper
├── pyproject.toml             # Project metadata and dependencies (managed via Poetry)
├── results/                   # Output directory for logs and experiment results
└── xgboost_reconstruction/    # Core implementation of the TimberStrike attack
```

> **Note:** The licenses for the included frameworks *FedTree* and *NVFlare* are provided in the `NOTICE` file.

## Gurobi License Configuration (Optional but recommended)

To enable the use of Gurobi for the optimization problem in this work, it is recommended to provide valid Gurobi credentials. This can be done by creating a `.env` file in the root directory with the following content:

```env
GUROBI_ACCESSID=<your_access_id>
GUROBI_SECRET=<your_secret_key>
GUROBI_LICENSEID=<your_license_id>
```

Make sure you have an active Gurobi license. These credentials are required to authenticate with the Gurobi Cloud or license server.

---

## 🐳 Using Docker

### Build the Docker Image

```bash
docker build -t timberstrike .
```

### Run the Docker Container

```bash
docker run --rm \
  --env-file .env \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  timberstrike ./run_stroke.sh
```

This command mounts the local `results/` directory into the container and executes the `run_stroke.sh` experiment script.

---

## 🛠️ Development Environment

TimberStrike uses [Poetry](https://python-poetry.org/) for dependency management.

### Install Dependencies

```bash
poetry install
```

### Optional: Enable Poetry Shell Plugin

If you are using a newer version of Poetry, the shell plugin may need to be added manually:

```bash
poetry self add poetry-plugin-poetry-shell
```

### Activate the Poetry Shell

```bash
poetry shell
```

### 🧱 Building FedTree

To build the FedTree library, refer to the instructions in the [FedTree](fl_systems/frameworks/FedTree/) directory. It contains the original README from the upstream project.

> **Note**: Some modules may require additional dependencies. Please consult the README files inside each respective subdirectory for detailed instructions.

### 🚀 Running Experiments

Use the following command to execute a complete experiment:

```bash
./run_all.sh <max_depth> <dataset_name>
```

Where:

- `<max_depth>` is the maximum depth of trees used in training.
- `<dataset_name>` is the dataset you wish to use (e.g., `stroke`, `diabetes`).

Configuration files are located in the [`config_yaml/`](config_yaml/) directory. These YAML files define:

- Federated learning settings (e.g., number of clients, rounds, trees)
- XGBoost hyperparameters
- Dataset partitioning strategies
- Evaluation tolerance

Ensure that the appropriate configuration is set before running an experiment.
You can find example configuration files in the `config_yaml/` folder to guide the creation of your own.

#### Using Your Own Datasets

To run experiments with a custom dataset, follow these steps:

1. **Add your dataset**:
   Place your dataset files inside a new folder under `dataset_partitioner/<your_dataset_name>/`. Preprocess the data as needed for your use case.

2. **Implement a dataloader**:
   In `dataset_partitioner/data_loader.py`, add a new function that loads your dataset and returns the feature matrix `X` and labels `y` for both training and test splits.

3. **Register your dataset**:
   Update the conditional logic at the beginning of `data_generator.py` to call your new dataloader function when your dataset name is specified.

4. **Define a configuration file**:
   Create a new YAML configuration file under `config_yaml/`, named as `<your_dataset_name>_<num_clients>.yaml`, to specify the desired experiment parameters.

5. **Run the experiment**:
   Use the provided script to execute your experiment:

   ```bash
   ./run_clients.sh <num_clients> <max_depth> <your_dataset_name>
   ```

---
For any further details, please refer to relevant module-specific READMEs, if available.
