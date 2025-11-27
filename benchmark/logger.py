import logging
import sys

def setup_logger(log_file: str):
    """Sets up the benchmark logger.

    Args:
        log_file (str): The path to the log file.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    # Get the root logger
    logger = logging.getLogger("vllm_benchmark")
    logger.setLevel(logging.INFO)

    # Prevent the root logger from propagating messages to the console
    logger.propagate = False

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- File Handler ---
    # Logs all levels (INFO, WARNING, ERROR) to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Console Handlers ---
    # Handler for INFO messages to stdout
    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno == logging.INFO)
    info_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    info_handler.setFormatter(info_formatter)
    logger.addHandler(info_handler)

    # Handler for WARNING and ERROR messages to stderr
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.WARNING) # Captures WARNING and ERROR
    error_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)
    
    return logger

# The logger will be initialized in main.py
# and can be imported from there by other modules.
# For modules that need a logger instance, they can do:
# import logging
# logger = logging.getLogger("vllm_benchmark")

# Convenience default logger reference (set after setup_logger is called).
logger = logging.getLogger("vllm_benchmark")
