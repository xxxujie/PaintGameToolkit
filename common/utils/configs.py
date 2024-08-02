import os
import yaml
import json
import cv2
import settings
from .loggers import get_logger

logger = get_logger()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        ext = os.path.splitext(config_path)[1]
        if ext == ".yaml":
            config = yaml.load(f, yaml.FullLoader)
        elif ext == ".json":
            config = json.load(f)
        else:
            raise ValueError("不支持的配置文件格式：" + ext)
        return config


def find_config_path(file_name: str):
    """按照 settings.CONFIG_DIRS 列表中的目录顺序查找名为 file_name（需要带扩展名）的配置文件"""

    for config_dir in settings.USER_CONFIG_DIRS:
        # print(f"Finding {file_name} in {config_dir}")
        config_path = os.path.join(config_dir, file_name)
        if os.path.exists(config_path):
            return config_path

    raise ValueError(f"找不到配置文件 {file_name}，请检查 CONFIG_DIRS 或者命令行参数")


class _Config:
    """所有配置类的基类，实际上是一个字典的包装类，具体见 _SampleConfig"""

    def __init__(self, config_path: str):
        self._config = load_config(config_path)

    def _get(self, *args):
        item = ""
        value = self._config
        for key in args:
            item += key
            if type(value) is not dict:
                raise ValueError(f"尝试从配置项 {item} 访问不存在的键 {key}")
            value = value.get(key)
            if value is None:
                logger.warning(f"配置项 {item} 为空")
                break
            item += "."

        return value


class _PBNConfig(_Config):
    @property
    def KMEANS_NCLUSTERS(self):
        return self._get("kmeans", "nclusters")

    @property
    def KMEANS_ATTEMPTS(self):
        return self._get("kmeans", "attempts")

    @property
    def KMEANS_CRITERIA_TYPE(self):
        type = self._get("kmeans", "criteria", "type")
        if not hasattr(self, "_kmeans_criteria_type"):
            tmp = 0
            if "TERM_CRITERIA_EPS" in type:
                tmp += cv2.TERM_CRITERIA_EPS
            if "TERM_CRITERIA_MAX_ITER" in type:
                tmp += cv2.TERM_CRITERIA_MAX_ITER
            if "TERM_CRITERIA_COUNT" in type:
                tmp += cv2.TERM_CRITERIA_COUNT
            setattr(self, "_kmeans_criteria_type", tmp)

        return getattr(self, "_kmeans_criteria_type")

    @property
    def KMEANS_CRITERIA_MAX_ITER(self):
        return self._get("kmeans", "criteria", "max_iter")

    @property
    def KMEANS_CRITERIA_EPSILON(self):
        return self._get("kmeans", "criteria", "epsilon")

    @property
    def KMEANS_FLAGS(self):
        flags = self._get("kmeans", "flags")
        if not hasattr(self, "_kmeans_flags"):
            if "KMEANS_PP_CENTERS" in flags:
                tmp = cv2.KMEANS_PP_CENTERS
            elif "KMEANS_RANDOM_CENTERS" in flags:
                tmp = cv2.KMEANS_RANDOM_CENTERS
            elif "KMEANS_INITIAL_LABELS" in flags:
                tmp = cv2.KMEANS_USE_INITIAL_LABELS
            else:
                raise ValueError("配置文件 pbn_conf 中的 flags 不是有效的值")
            setattr(self, "_kmeans_flags", tmp)

        return getattr(self, "_kmeans_flags")

    @property
    def MIN_AREA(self):
        return self._get("min_area")

    @property
    def SHOW_BOTTOM_PANEL(self):
        return self._get("show_bottom_panel")

    @property
    def PANEL_HEIGHT(self):
        return self._get("panel_height")

    @property
    def CONTOUR_RETRIEVAL_MODE(self):
        if not hasattr(self, "_contour_retrieval_mode"):
            match self._get("contour", "retrieval_mode"):
                case "RETR_EXTERNAL":
                    tmp = cv2.RETR_EXTERNAL
                case "RETR_LIST":
                    tmp = cv2.RETR_LIST
                case "RETR_CCOMP":
                    tmp = cv2.RETR_CCOMP
                case "RETR_TREE":
                    tmp = cv2.RETR_TREE
                case _:
                    raise ValueError(
                        "配置文件 pbn_conf 中的 contour.retrieval_mode 不是有效的值"
                    )
            setattr(self, "_contour_retrieval_mode", tmp)

        return getattr(self, "_contour_retrieval_mode")

    @property
    def CONTOUR_APPROX_MODE(self):
        if not hasattr(self, "_contour_approx_mode"):
            match self._get("contour", "approx_mode"):
                case "CHAIN_APPROX_NONE":
                    tmp = cv2.CHAIN_APPROX_NONE
                case "CHAIN_APPROX_SIMPLE":
                    tmp = cv2.CHAIN_APPROX_SIMPLE
                case _:
                    raise ValueError(
                        "配置文件 pbn_conf 中的 contour.approx_mode 不是有效的值"
                    )
            setattr(self, "_contour_approx_mode", tmp)

        return getattr(self, "_contour_approx_mode")

    @property
    def SLIC_REGION_SIZE(self):
        return self._get("slic", "region_size")

    @property
    def SLIC_ALGORITHM(self):
        if not hasattr(self, "_slic_algorithm"):
            match self._get("slic", "algorithm"):
                case "SLIC":
                    tmp = cv2.ximgproc.SLIC
                case "SLICO":
                    tmp = cv2.ximgproc.SLICO
                case "MSLIC":
                    tmp = cv2.ximgproc.MSLIC
                case _:
                    raise ValueError("配置文件 pbn_conf 中的 slic.algorithm 不是有效的值")
            setattr(self, "_slic_algorithm", tmp)

        return getattr(self, "_slic_algorithm")

    @property
    def SLIC_NUM_ITERATIONS(self):
        return self._get("slic", "num_iterations")

    @property
    def SLIC_GAUSSIAN_KSIZE(self):
        return self._get("slic", "gaussian_blur", "ksize")

    @property
    def SLIC_GAUSSIAN_SIGMA_X(self):
        return self._get("slic", "gaussian_blur", "sigmaX")

    @property
    def SLIC_GAUSSIAN_SIGMA_Y(self):
        return self._get("slic", "gaussian_blur", "sigmaY")


# 留给外部调用的单例，初始化需要指定对应配置文件的地址
pbn_config = _PBNConfig(find_config_path("pbn_conf.yaml"))
