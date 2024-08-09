import os
import yaml
import json

from .settings import tool_settings


def load_config(config_path: str, encoding="utf-8") -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding=encoding) as f:
        ext = os.path.splitext(config_path)[1]
        if ext == ".yaml":
            config = yaml.load(f, yaml.SafeLoader)
        elif ext == ".json":
            config = json.load(f)
        else:
            raise ValueError("不支持的配置文件格式：" + ext)
        return config


def find_config_path(file_name: str):
    """按照 settings.CONFIG_DIRS 列表中的目录顺序查找名为 file_name（需要带扩展名）的配置文件"""

    for config_dir in tool_settings.CONFIG_DIRS:
        # print(f"Finding {file_name} from {config_dir}")
        config_path = os.path.join(config_dir, file_name)
        if os.path.exists(config_path):
            return config_path

    raise ValueError(
        f"找不到配置文件 {file_name}，请检查 CONFIG_DIRS 中是否存在该配置文件"
    )
