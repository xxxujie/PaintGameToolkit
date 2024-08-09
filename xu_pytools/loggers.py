import os
import yaml
import logging
import logging.config

from .settings import tool_settings


class _Logger:
    def __init__(self, name, outfile_dir):
        # 保证目录存在并能够读写
        if not os.path.exists(outfile_dir):
            os.mkdir(outfile_dir, mode=0o766)
        os.chmod(outfile_dir, 0o766)

        self._outfile_dir = outfile_dir
        self._name = name
        self.logger = self._init_logger()

    def _init_logger(self):
        # 加载配置文件，读取输出文件名拼成绝对路径
        with open(tool_settings.LOGGERS_CONF_PATH, "r", encoding="utf-8") as f:
            config = yaml.load(f, yaml.SafeLoader)
        fname = config["handlers"]["info_file"]["filename"]
        config["handlers"]["info_file"]["filename"] = os.path.join(
            self._outfile_dir, fname
        )
        fname = config["handlers"]["error_file"]["filename"]
        config["handlers"]["error_file"]["filename"] = os.path.join(
            self._outfile_dir, fname
        )
        logging.config.dictConfig(config)
        logger = logging.getLogger(self._name)
        return logger


def get_logger(name: str = "default", outfile_dir: str = tool_settings.LOG_OUTFILES_DIR):
    logger = _Logger(name, outfile_dir).logger
    return logger
