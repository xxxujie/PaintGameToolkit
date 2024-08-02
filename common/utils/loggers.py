import logging
import logging.config
import os
import yaml
import json
import settings


class _Logger:
    def __init__(self, name, log_dir=settings.LOG_DIR):
        self._log_dir = log_dir
        self._name = name
        self.logger = self._init_logger()

    def _init_logger(self):
        with open(settings.LOGGING_CONFIG_PATH, "r", encoding="utf-8") as f:
            ext = os.path.splitext(settings.LOGGING_CONFIG_PATH)[1]
            if ext == ".yaml":
                config = yaml.load(f, yaml.SafeLoader)
            elif ext == ".json":
                config = json.load(f)
            else:
                raise ValueError("不支持的 logging 配置文件格式：" + ext)
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
