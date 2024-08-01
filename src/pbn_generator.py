import os
import cv2
import numpy as np
import settings

from cv2.typing import MatLike
from common.utils.configs import pbn_config
from common.utils.loggers import MyLogger


logger = MyLogger(__name__)


def img2pbn(src_path: str):
    """将原图转换成 paint-by-numbers 风格的图片,
    如果 src_path 是一个目录，则会转换目录下所有图片；
    如果 src_path 是一个文件路径，则转换指定图片。

    Args:
        src_path (str): 原图所在目录或单张图片路径
        tgt_dir (str): PBN 图片存放目录
    """
    if os.path.isfile(src_path):
        _convert_and_save(src_path)
    elif os.path.isdir(src_path):
        img_paths = os.listdir(src_path)
        for img_path in img_paths:
            _convert_and_save(os.path.join(src_path, img_path))


def _save_img(img: MatLike, img_name, tag=""):
    name_without_ext = os.path.splitext(img_name)[0]
    name_ext = os.path.splitext(img_name)[1]
    if not os.path.exists(settings.OUTPUT_DIR):
        os.makedirs(settings.OUTPUT_DIR)
    cv2.imwrite(os.path.join(settings.OUTPUT_DIR, name_without_ext + tag + name_ext), img)


def _convert_and_save(img_path: str):
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    # 先通过聚类分离原图区域
    recolored_img, area_parts, centers = _clusterize(img)
    # 再画出轮廓图
    pbn_img = _draw_outline(recolored_img.shape[:2], area_parts, centers)

    # idx = 0
    # for part in area_parts:
    #     _save_img(part, img_name, f"_part{idx}")
    #     idx += 1

    _save_img(pbn_img, img_name, "_pbn")


def _clusterize(img: MatLike) -> tuple[MatLike, list[MatLike], MatLike]:
    """K-Mean 算法按区域分割图像

    Args:
        img_path (str): 图像路径

    Returns:
        tuple[MatLike, list[MatLike], MatLike]: 分别为 (重新上色后的图像, 各标签对应的区域二值图像, 簇心集合)
    """
    data = img.reshape((-1, 3))
    criteria = (
        pbn_config.KMEANS_CRITERIA_TYPE,
        pbn_config.KMEANS_CRITERIA_MAX_ITER,
        pbn_config.KMEANS_CRITERIA_EPSILON,
    )

    compactness, labels, centers = cv2.kmeans(
        data.astype(np.float32),
        pbn_config.KMEANS_NCLUSTERS,
        None,
        criteria,
        pbn_config.KMEANS_ATTEMPTS,
        pbn_config.KMEANS_FLAGS,
    )
    # 同一簇的像素点全部使用中心的颜色
    labels = labels.flatten()
    recolored_img = centers[labels].reshape(img.shape)
    # 分别分离每个标签对应的图像部分
    area_parts = []
    for label in range(centers.shape[0]):
        # part 是每个部分的图像，不需要通道维度，一开始是全黑，先展成一维以便计算
        part = np.zeros(img.shape[:2], np.uint8).reshape((-1, 1))
        for pixel_idx in range(part.size):
            # 如果该位置所属标签和当前标签相同，则该位置的像素设为白色（255)
            if labels[pixel_idx] == label:
                part[pixel_idx] = 255
        part = part.reshape(img.shape[:2])
        area_parts.append(part)

    return recolored_img, area_parts, centers


def _draw_outline(
    img_shape: tuple[int, int], area_parts: list[MatLike], centers: MatLike
):
    """根据多个区域二值图和簇心画出轮廓图

    Args:
        img_shape (Shape): 二值图像尺寸
        area_parts (list[MatLike]): 各标签对应的区域图像数组
        centers (MatLike): K-Means 获得的簇心集合

    Returns:
        MatLike: 画出轮廓后的图像
    """
    # 过滤区域并标号
    filtered_contours = []
    for part in area_parts:
        # findContours 就是找黑底图的白色对象
        contours, hierarchy = cv2.findContours(
            part, pbn_config.CONTOUR_RETRIEVAL_MODE, pbn_config.CONTOUR_APPROX_MODE
        )
        # 过滤面积小于 min_area 的区域
        for contour in contours:
            if cv2.contourArea(contour) > pbn_config.MIN_AREA:
                filtered_contours.append(contour)

        # 在区域中标号
        # for k, contour in enumerate(filtered_contours):
        #     if hierarchy[0, k, 3] < 0:
        #         cv2.putText(
        #             black_img,
        #             str(part_idx),
        #             (int(contour[0, 0, 0]), int(contour[0, 0, 1]) + 14),
        #             cv2.FONT_HERSHEY_PLAIN,
        #             1,
        #             (100, 100, 100),
        #         )

    # 白底轮廓图，颜色用 200 灰色模仿手绘风格
    contour_img = np.ones(img_shape, dtype=np.uint8) * 255
    cv2.drawContours(contour_img, filtered_contours, -1, (200, 200, 200))

    # 绘制图像底部的颜色展示面板
    bott_panel = np.zeros(
        (pbn_config.PANEL_HEIGHT, contour_img.shape[1], 3), dtype=np.uint8
    )
    average_width = bott_panel.shape[1] // centers.shape[0]
    rect_width = average_width - 20
    # 创建矩形
    for center_idx in range(centers.shape[0]):
        col1, col2, col3 = centers[center_idx]
        cv2.putText(
            bott_panel,
            f"{center_idx}:",
            (10 + center_idx * average_width, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
        )
        cv2.rectangle(
            bott_panel,
            (58 + center_idx * average_width, 5),
            (rect_width + center_idx * average_width, 45),
            (255, 255, 255),
        )
        cv2.rectangle(
            bott_panel,
            (59 + center_idx * average_width, 6),
            (rect_width - 1 + center_idx * average_width, 44),
            (int(col1), int(col2), int(col3)),
            -1,
        )

    # 将轮廓二值图（当成灰度图）  转换为 BGR 图像
    pbn_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
    if pbn_config.SHOW_BOTTOM_PANEL:
        # 拼接轮廓图和颜色面板
        pbn_img = np.vstack((pbn_img, bott_panel))

    return pbn_img
