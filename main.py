import argparse
import settings

from src import pbn_generator
from common.utils import loggers


def main(args):
    if args.config_dir != "":
        settings.CONFIG_DIRS.insert(0, args.config_dir)
    logger = loggers.get_logger()
    logger.info("Start running")


if __name__ == "__main__":
    # 定义一个解析器，用于命令行执行时解析附带的参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="./resources/configs")

    args = parser.parse_args()

    main(args)

    pbn_generator.img2pbn(settings.DATA_DIR)
