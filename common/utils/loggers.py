import os
import logging
import settings


class MyLogger:

    def __init__(
        self,
        name,
        log_path=os.path.join(settings.LOG_DIR, "app.log"),
        log_level=logging.INFO,
    ):
        self._name = name
        self._log_path = log_path
        self._log_level = log_level
        self._logger = self._init_logger()

    def _init_logger(self):
        logger = logging.getLogger(self._name)
        logger.setLevel(self._log_level)
        logger_formatter = logging.Formatter(
            fmt=(
                "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] [P%(process)d:%(threadName)s] "
                "%(message)s"
            ),
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        # 创建StreamHandler输出到控制台
        logger_console_handler = logging.StreamHandler()
        logger_console_handler.setLevel(logging.INFO)
        logger_console_handler.setFormatter(logger_formatter)
        # 创建FileHandler输出到文件
        logger_file_handler = logging.FileHandler(filename=self._log_path, mode="a")
        logger_file_handler.setLevel(logging.DEBUG)
        logger_file_handler.setFormatter(logger_formatter)
        # 将Handler添加到Logger
        logger.addHandler(logger_console_handler)
        logger.addHandler(logger_file_handler)
        return logger

    def get_logger(self):
        return self._logger

    def debug(self, msg, *args):
        self._logger.debug(msg, *args)

    def info(self, msg, *args):
        self._logger.info(msg, *args)

    def warning(self, msg, *args):
        self._logger.warning(msg, *args)

    def error(self, msg, *args):
        self._logger.error(msg, *args)

    def exception(self, msg, *args):
        self._logger.exception(msg, *args)
