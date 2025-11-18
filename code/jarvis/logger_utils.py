# logger_utils.py
import logging
import os

def setup_logger(config: dict) -> logging.Logger:
    """
    Setup a logger based on config.toml settings.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from TOML.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_cfg = config.get("logging", {})
    enable = log_cfg.get("enable", True)
    level = log_cfg.get("level", "INFO").upper()
    log_file = log_cfg.get("file", "project.log")

    logger = logging.getLogger("jarvis_project")
    logger.setLevel(getattr(logging, level, logging.INFO))

    if enable:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, level, logging.INFO))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level, logging.INFO))

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Attach handlers if not already
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(ch)

    return logger

def flush_logger(logger: logging.Logger) -> None:
    """
    Flush all handlers attached to the given logger.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance created by setup_logger.
    """
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception as e:
            logger.error(f"Failed to flush handler {handler}: {e}")
