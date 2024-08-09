from os.path import dirname, abspath


# Base directory
BASE_DIR = dirname(dirname(dirname(abspath(__file__))))

# Where user's configuration files locate (finding by order)
CONFIG_DIRS = [
    BASE_DIR,
    f"{BASE_DIR}/configs",
]

# Where logs locate
LOG_OUTFILES_DIR = f"{BASE_DIR}/logs"

LOGGERS_CONF_PATH = f"{BASE_DIR}/xu_pytools/settings/loggers_setting.yaml"
