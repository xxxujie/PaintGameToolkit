import logging
import logging.config
import os
import settings
from .configs import load_config, find_config_path


class _Logger:
    def __init__(self, name, log_dir=settings.LOG_DIR):
        self._log_dir = log_dir
        self._name = name
        self.logger = self._init_logger()

    def _init_logger(self):
        config = load_config(find_config_path("logging_conf.yaml"))
        fname = config["handlers"]["info_file"]["filename"]
        config["handlers"]["info_file"]["filename"] = os.path.join(self._log_dir, fname)
        fname = config["handlers"]["error_file"]["filename"]
        config["handlers"]["error_file"]["filename"] = os.path.join(self._log_dir, fname)
        logging.config.dictConfig(config)
        logger = logging.getLogger(self._name)
        return logger


def get_logger(name="default"):
    logger = _Logger(name).logger
    return logger
