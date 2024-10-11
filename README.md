## Quick Start

### 准备需要的素材图片

准备好的图片放在 resources/data 目录下，可以放任意张，分割器会遍历目录下所有图片进行分割。
   
- 尺寸要求：无。不过涂色游戏项目需要图像长和宽小于等于 1024 像素。
- 格式要求：所有 cv2.imread() 支持的格式都支持。
  - 但是为了涂色游戏的效果，.png是最佳选择，因为它会保留透明区域。

### 调整参数

参数配置文件位置为 configs/pbn_config.yaml。每一部分都有注释说明，需要特别注意的是：

1. superpixel_algorithm 用于调整超像素的算法选择，包括 SLIC 和 SEED 两类，下面的 slic 和 seed 部分分别是两种算法需要的参数。
2. kmeans.nclusters 用于调整聚类数目，实际上就是分割色块的数目，比如 60 就是会分割出 60 种颜色。

```yaml
# 超像素前的高斯模糊操作
gaussian_blur:
  # 高斯核尺寸
  ksize: [5, 5]
  sigmaX: 0
  sigmaY: 0

# 超像素算法，可以是 SLIC 或 SEED
superpixel_algorithm: "SEED"

slic:
  # 超像素大小，决定了超像素数量（= 原像素数量 / 超像素大小）
  region_size: 32
  # 使用哪种 SLIC 算法变体，包括：SLIC、SLICO（默认）、MSLIC
  algorithm: SLICO
  # SLIC 迭代次数
  num_iterations: 100

seed:
  # 超像素数量
  num_superpixels: 5000
  num_levels: 30 
  prior: 5
  histogram_bins: 5
  # SEED 迭代次数
  num_iterations: 100

kmeans:
  # 聚类数目（K）
  nclusters: 60

  # 尝试次数，使用结果最好的一次
  attempts: 10

  # 终止条件
  criteria:
    # 终止条件类型，包括：
    # 1. TERM_CRITERIA_EPS：簇心变动幅度低于 epsilon 时终止
    # 2. TERM_CRITERIA_MAX_ITER：达到最大迭代次数时终止
    # 可叠加 - "TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER"
    type: TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER
    # 最大迭代次数
    max_iter: 1000
    # 簇心变动幅度
    epsilon: 0.01

  # K-Means 初始化簇心的方法，包括：
  # 1. KMEANS_PP_CENTERS：按照 K-Means++ 论文中的方法
  # 2. KMEANS_RANDOM_CENTERS：随机初始化
  flags: KMEANS_PP_CENTERS

contour:
  # 提取轮廓层级的模式，包括：
  # 1. RETR_EXTERNAL：只提取最外部轮廓
  # 2. RETR_LIST：提取所有轮廓，但不建立父子和嵌套关系
  # 3. RETR_CCOMP：提取所有轮廓，并组织为两个层级的关系
  # 4. RETR_TREE：提取所有轮廓，并包含完成的层级关系
  retrieval_mode: RETR_TREE
  # 提取轮廓点的模式，包括：
  # 1. CHAIN_APPROX_NONE：存储所有轮廓点
  # 2. CHAIN_APPROX_SIMPLE：只存储必要的轮廓点以节省空间，比如矩形只需要四个顶点
  approx_mode: CHAIN_APPROX_SIMPLE
  # 轮廓线粗细
  line_thickness: 1

# 限制最小区域尺寸，低于该尺寸的区域将被过滤
# ! 设置为 0 就行，因为过滤掉一些区域会导致游戏里缺一些色块
min_area: 16

# 轮廓线颜色
contour_color: [50, 50, 50]

# 是否在区域内写上标号
show_area_index: false
```

### 运行代码

在项目根目录下，命令行执行：`python main.py`

### 输出结果
   
输出的结果保存在 resources/outputs 文件夹下。

![image](https://github.com/user-attachments/assets/5886d6a1-81a0-4b2a-8b8e-bbb65a570b96)

- xxx_area_infos.json：区域信息，包括每一个分割区域的颜色 ID 和位置信息。
- xxx_color_infos.json：颜色信息，包括每一个颜色 ID 对应的 RGB 值。
  - 提醒！Unity 中的 RGB 值是 0~2.55 而不是 0~255，所以在 Unity 里使用的时候要先除以 100。
- xxx_contours.png：一张只有轮廓的图。
  - 该图在涂色游戏中暂时没用到，如果需要带轮廓风格的样式，可以把这张图叠在最上层。
- xxx_part_<n>.png：每一部分的区域图，已经灰度化。

## 代码详解

需要理解和修改代码的话，阅读下面章节。

### 目录结构

![image](https://github.com/user-attachments/assets/db0003e3-cf58-4487-bc77-fa55feb63503)

- configs：配置文件，可以是 .yaml 或 .json 文件。
- logs：输出日志，当天的日志为 app.log，每天 0 点自动备份名如 app.log.2024-10-11 的文件。
- xu_pytools：Python 工具，包括日志工具和 config 工具。使用方法见下面章节。
- resources：图片的输入和输出目录，需要分割的图片放在 resources/data 下，分割后的输出将在 resources/outputs 下。
- src：源码目录，包括一个 pbn_generator.py 文件。
- main.py：入口代码文件，运行改代码即可启动分割。
  - python ./main.py
- requirements.txt：项目需要的依赖包，命令行使用 pip install -r requirements.txt 来安装。
- Dockerfile：Docker 镜像打包文件（不使用 Docker 的话不需要管）。

### xu_pytools: 日志工具

`xu_pytools/loggers.py`

#### 日志的使用

```python
# 导入日志模块
from xu_pytools import loggers

# 在导包后面，其他所有代码之前，获取日志记录器
logger = loggers.get_looger()

# 之后就可以使用 logger 进行日志打印了
# 打印的日志既会输出到终端，又会输出到日志文件中
logger.debug("hello");
logger.info("hello");
logger.warning("hello");
# error 不带跟踪栈信息，exception 带跟踪栈信息
logger.error("hello");
logger.exception("hello");
```

#### 日志的配置

日志的配置文件在 `xu_pytools/settings/loggers_setting.yaml`，本日志工具是基于 logging 包写的，配置文件也和它一样，具体配置详见：[https://docs.python.org/zh-cn/3/library/logging.config.html](https://docs.python.org/zh-cn/3/library/logging.config.html)。

### xu_pytools: 配置文件工具

`xu_pytools/configs.py`

配置文件工具可以将外部配置文件（yaml 和 json）转化为 Python 类的形式，从而方便开发者使用。

#### 配置文件工具的使用

1. 配置文件目录
   
   该工具会以 xu_pytools 文件夹的父目录为根目录，读取所有 <根目录>/ 和 <根目录>/configs/ 目录中的所有 .yaml 和 .json 配置文件。

2. 继承 Config 基类，编写自定义 Config 类
   
   直接写在 config.py 中，也可以重新写新文件，只要注意在代码中导入正确的模块即可。

3. 为自定义 Config 类添加需要的属性
   
   使用装饰器 @property 就可以为 Python 类添加 Getter 属性，并利用基类的 self._get(*args) 来获取配置文件中的字段。

4. 初始化一个供外部使用的实例

   使用 my_config = MyConfig("my_config.yaml")。传入参数为配置文件全名。

#### 举例

假如我们的配置文件有以下字段：

```yaml
# <BASE_DIR>/configs/character_config.yaml
name: "Bob"
age: 20
outlook:
  hair_color: "black"
  wearing_glass: true
```

那么我们需要自定义 Config 类：

```python
class MyCharacterConfig(_Config):
    @property
    def NAME(self):
        return self._get("name")
    
    @property
    def AGE(self):
        return self._get("age")
        
    @property
    def OUTLOOK_HAIR_COCLOR(self):
        return self._get("outlook", "hair_color")
    
    @property
    def OUTLOOK_WEARING_GLASS(self):
        return self._get("outlook", "wearing_glass")
        
character_config = MyCharacterConfig("character_config.yaml");
```

于是外部就可以通过 `character_config` 来读取配置文件了。

```python
print(f"My name is {character_config.NAME}")
```
