from pathlib import Path

import logging


def build_logger(
    log_dir,
    log_name="vision_training.log",
):
    """
    Build a reusable logger.

    Logs are written both to:

        console

    and

        artifacts/logs/
    """

    log_dir = Path(log_dir)

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "VisionTrainer"
    )

    logger.setLevel(
        logging.INFO
    )

    ####################################################
    # Avoid duplicate handlers
    ####################################################

    if logger.hasHandlers():

        logger.handlers.clear()

    ####################################################
    # Formatter
    ####################################################

    formatter = logging.Formatter(

        "[%(asctime)s] "

        "%(levelname)s "

        "%(message)s",

        datefmt="%Y-%m-%d %H:%M:%S",

    )

    ####################################################
    # Console
    ####################################################

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    ####################################################
    # File
    ####################################################

    file_handler = logging.FileHandler(

        log_dir / log_name,

        mode="a",

    )

    file_handler.setFormatter(
        formatter
    )

    ####################################################
    # Register
    ####################################################

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    return logger