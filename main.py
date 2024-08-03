import argparse
import settings

from src import pbn_generator
from common.utils import loggers


def main():
    logger.info("Start running")
    pbn_generator.img2pbn(settings.DATA_DIR)


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="")
    args = parser.parse_args()
    if args.config_dir != "":
        settings.USER_CONFIG_DIRS.insert(0, args.config_dir)
    logger = loggers.get_logger()
    main()
