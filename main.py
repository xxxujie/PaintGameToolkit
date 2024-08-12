from src import pbn_generator
from xu_pytools import loggers, configs


def main():
    logger.info("Start running")
    pbn_generator.img2pbn(configs.app_config.DATA_DIR)


if __name__ == "__main__":
    logger = loggers.get_logger()
    main()
