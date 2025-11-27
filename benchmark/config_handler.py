import yaml

def load_config(config_path="bench_config.yaml"):
    """Loads the benchmark configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
