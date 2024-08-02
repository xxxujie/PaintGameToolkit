import os
import yaml
import json
import cv2
import settings


class _ConfigHandler:
    """(Deprecated)
    读取所有配置文件，自动添加字典类型的配置实例作为属性
    """

    def __init__(self, config_dir):
        self.reload(config_dir)

    def reload(self, config_dir):
        loader = self._load_configs(config_dir)
        # 根据配置文件自动添加字典属性
        for name, config in loader:
            setattr(self, name, config)

    def _load_configs(self, config_dir: str):
        """加载配置文件"""
        for fname in os.listdir(config_dir):
            with open(os.path.join(config_dir, fname), "r", encoding="utf-8") as f:
                ext = os.path.splitext(fname)[1]
                if ext == ".yaml":
                    config = yaml.load(f, yaml.FullLoader)
                elif ext == ".json":
                    config = json.load(f)
                else:
                    raise ValueError("不支持的配置文件格式：" + ext)
                yield os.path.splitext(fname)[0], dict(config)


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

    for config_dir in settings.CONFIG_DIRS:
        # print(f"Finding {file_name} in {config_dir}")
        config_path = os.path.join(config_dir, file_name)
        if os.path.exists(config_path):
            return config_path

    raise ValueError(f"找不到配置文件 {file_name}，请检查 CONFIG_DIRS 或者命令行参数")


class _Config:
    """所有配置类的基类，实际上是一个字典的包装类，具体见 _SampleConfig"""

    def __init__(self, config_path: str):
        self._config = load_config(config_path)

    def get(self, key):
        return self._config.get(key)


class _PBNConfig(_Config):
    @property
    def KMEANS_NCLUSTERS(self):
        return self.get("kmeans").get("nclusters")

    @property
    def KMEANS_ATTEMPTS(self):
        return self.get("kmeans").get("attempts")

    @property
    def KMEANS_CRITERIA_TYPE(self):
        type = self.get("kmeans").get("criteria").get("type")
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
        return self.get("kmeans").get("criteria").get("max_iter")

    @property
    def KMEANS_CRITERIA_EPSILON(self):
        return self.get("kmeans").get("criteria").get("epsilon")

    @property
    def KMEANS_FLAGS(self):
        flags = self.get("kmeans").get("flags")
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
        return self.get("min_area")

    @property
    def SHOW_BOTTOM_PANEL(self):
        return self.get("show_bottom_panel")

    @property
    def PANEL_HEIGHT(self):
        return self.get("panel_height")

    @property
    def CONTOUR_RETRIEVAL_MODE(self):
        if not hasattr(self, "_contour_retrieval_mode"):
            match self.get("contour").get("retrieval_mode"):
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
            match self.get("contour").get("approx_mode"):
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
        return self.get("slic").get("region_size")

    @property
    def SLIC_ALGORITHM(self):
        if not hasattr(self, "_slic_algorithm"):
            match self.get("slic").get("algorithm"):
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
        return self.get("slic").get("num_iterations")

    @property
    def SLIC_GAUSSIAN_KSIZE(self):
        return self.get("slic").get("gaussian_blur").get("ksize")

    @property
    def SLIC_GAUSSIAN_SIGMA_X(self):
        return self.get("slic").get("gaussian_blur").get("sigmaX")

    @property
    def SLIC_GAUSSIAN_SIGMA_Y(self):
        return self.get("slic").get("gaussian_blur").get("sigmaY")


# 留给外部调用的单例，初始化需要指定对应配置文件的地址
pbn_config = _PBNConfig(find_config_path("pbn_conf.yaml"))
