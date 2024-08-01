# Paint Game Toolkit

开发涂色游戏的工具套件，包括一个将图像转换为 PBN 风格的工具 `src.pbn_generator.py`。

## 配置文件

默认情况下，会在 `./resources/configs` 下查找需要的配置文件，可修改 `settings.py` 中的 `CONFIG_DIR` 以更改查找路径。

如果在运行 `main.py` 时指定了参数 `--config_dir`，则优先使用参数指定的配置目录。