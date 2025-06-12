from vdapa.config import config, BASE_DIR
import logging
from pathlib import Path


def setup_logging(subpackage, module):
    """
    Sets up a logger that writes to ./logs/<subpackage>/<module>.log
    and also logs to console.

    Parameters:
        subpackage (str): The subpackage name, used as a folder inside logs.
        module (str): The module name, used as the log filename and logger name.

    Returns:
        logging.Logger: Configured logger instance.
    """

    log_dir = BASE_DIR / Path (config['paths']['logs']) / subpackage
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{module}.log"

    logger = logging.getLogger(f"{subpackage}.{module}")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
