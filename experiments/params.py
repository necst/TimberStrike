import yaml

from experiments.tree_based_utils import PartitionType

# Define default values for the configuration
DEFAULT_PARAMS = {
    'num_clients': 5,
    'num_rounds': 10,
    'force_partition': False,
    'force_training': False,
    'force_attack': False,
    'xgb_param': {
        'eta': 0.3,
        'gamma': 0,
        'max_depth': 6,
        'lambda': 1,
    },
    'baseline': False,
    'log_dir': "results",
    'data': "stroke",
    'data_path': "./dataset_partitioner/healthcare-dataset-stroke-prep.csv",
    'niid_type': PartitionType.IID,
    'skip': False,
    'alpha': 0.5,
    'victim_cid': 0,
    'tolerance': 0.1,
    'to_drop': [],
}


# Helper function to load and parse the YAML file
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config or {}


# Helper function to deeply merge dictionaries
def deep_merge_dicts(base, overrides):
    merged = base.copy()
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


# Main function to parse the entire configuration
def experiments_config_parser(yaml_file_path=None):
    # Load YAML file if provided, otherwise start with an empty config
    yaml_config = load_yaml_config(yaml_file_path)

    # Start with global defaults and merge in any global overrides from YAML
    global_params = deep_merge_dicts(DEFAULT_PARAMS, {k: v for k, v in yaml_config.items() if
                                                      k not in ['fedxgbllr', 'nvflare', 'cyclic', 'bagging',
                                                                'fedtree']})

    # Define each system's configuration by merging global defaults and system-specific YAML overrides
    systems = ['fedxgbllr', 'nvflare', 'cyclic', 'bagging', 'fedtree']
    systems_config = {}

    for system in systems:
        # Start with a copy of the global params for each system
        system_config = deep_merge_dicts(global_params, yaml_config.get(system, {}))

        # If system-specific `xgb_param` exists, fully merge it with the global `xgb_param`
        if 'xgb_param' in yaml_config.get(system, {}):
            system_config['xgb_param'] = deep_merge_dicts(
                global_params['xgb_param'], yaml_config[system]['xgb_param']
            )

        # Assign the merged configuration to the system's key
        systems_config[system] = system_config

    # Combine global and system-specific configurations into a single dictionary without duplicates
    final_config = {
        'global': global_params,
        **systems_config
    }

    return final_config


# Display function for testing
def display_config(config):
    print("Global Parameters:")
    print(config['global'])
    print("\nSystem-Specific Configurations:")
    for system, system_config in config.items():
        if system != 'global':
            print(f"{system}: {system_config}")
