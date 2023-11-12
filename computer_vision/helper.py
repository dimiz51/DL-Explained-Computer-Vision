import json
import os

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load config function
def load_config(model_name: str) -> dict:
    config_path = os.path.join(script_dir, 'configs.json')
    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    # Check if the model_name is present in the configurations
    if model_name in all_configs:
        return all_configs[model_name]
    else:
        raise ValueError(f"Configuration not found for model: {model_name}")
