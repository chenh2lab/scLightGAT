import logging
import sys

def setup_logger(name: str = "scLightGAT"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add formatter to console handler
        ch.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(ch)

    return logger
