import logging
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
            logging.getLogger("vllm_benchmark").info(f"Loading configuration from '{config_path}'")
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.getLogger("vllm_benchmark").error(f"Configuration file not found at '{config_path}'")
        return None
    except yaml.YAMLError as e:
        logging.getLogger("vllm_benchmark").error(f"Error parsing YAML file: {e}")
        return None
