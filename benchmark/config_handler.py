import yaml
from logger import logger

def load_config(config_path="bench_config.yaml"):
    """Loads the benchmark configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration.
    """
    try:
        with open(config_path, "r") as f:
            logger.info(f"Loading configuration from '{config_path}'")
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at '{config_path}'")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return None
