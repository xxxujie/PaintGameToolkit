import os
import cv2
import numpy as np
import settings

from tqdm import tqdm
from cv2.typing import MatLike
from common.utils.configs import pbn_config
from common.utils import loggers


logger = loggers.get_logger()


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
    saved_path = os.path.join(settings.OUTPUT_DIR, name_without_ext + tag + name_ext)
    cv2.imwrite(saved_path, img)
    return saved_path


def _convert_and_save(img_path: str):
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    logger.info(f"开始转换 PBN（for {img_name}）")
    # 先通过聚类分离原图区域
    # slic_img, recolored_img, area_parts, centers = _clusterize(img)
    slic_img, recolored_img, area_parts, centers = _cluster_with_superpixel(
        img, pbn_config.SUPERPIXEL_ALGORITHM
    )
    # 再画出轮廓图
    pbn_img = _draw_outline(recolored_img.shape[:2], area_parts, centers)

    canny_img = _canny(img)

    # idx = 0
    # for part in area_parts:
    #     _save_img(part, img_name, f"_part{idx}")
    #     idx += 1

    _save_img(slic_img, img_name, "_superpixel")
    _save_img(recolored_img, img_name, "_recolored")
    _save_img(canny_img, img_name, "_canny")
    saved_path = _save_img(pbn_img, img_name, "_pbn")
    logger.info(f"转换完成！（saved in {saved_path}）")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
    custom_img = np.ones(img_gray.shape, dtype=np.uint8) * 255
    for idx in range(img_gray.size):
        if img_gray[idx] < 100:
            custom_img[idx] = 0
    custom_img = custom_img.reshape(img.shape[:2])
    custom_img = cv2.cvtColor(custom_img, cv2.COLOR_GRAY2BGR)
    _save_img(custom_img, img_name, "_custom")


def _canny(img):
    canny_img = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    canny_img = cv2.bitwise_not(canny_img)
    return canny_img


def _cluster_with_superpixel(img: MatLike, sp_algorithm):
    # 先经过 superpixel 拿到超像素特征图
    # 1. 高斯模糊
    img = cv2.GaussianBlur(
        img,
        ksize=pbn_config.GAUSSIAN_KSIZE,
        sigmaX=pbn_config.GAUSSIAN_SIGMA_X,
        sigmaY=pbn_config.GAUSSIAN_SIGMA_Y,
    )
    # 2. 转换到LAB颜色空间
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # 3. 创建 SLIC/SEED 超像素对象并迭代分割
    if sp_algorithm == "SLIC":
        logger.info(
            f"正在进行超像素计算（Algo - SLIC, "
            f"RegionSize - {pbn_config.SLIC_REGION_SIZE}, "
            f"NumIters - {pbn_config.SLIC_NUM_ITERATIONS}）"
        )
        sp = cv2.ximgproc.createSuperpixelSLIC(
            lab_img,
            algorithm=pbn_config.SLIC_ALGORITHM,
            region_size=pbn_config.SLIC_REGION_SIZE,
        )
        sp.iterate(pbn_config.SLIC_NUM_ITERATIONS)
    elif sp_algorithm == "SEED":
        logger.info(
            f"正在进行超像素计算（Algo - SEED, "
            f"NumSuperpixels - {pbn_config.SEED_NUM_SUPERPIXELS}, "
            f"NumLevels - {pbn_config.SEED_NUM_LEVELS}, "
            f"Prior - {pbn_config.SEED_PRIOR}, "
            f"Histogram bins - {pbn_config.SEED_HISTOGRAM_BINS}, "
            f"NumIters - {pbn_config.SEED_NUM_ITERATIONS}）"
        )
        sp = cv2.ximgproc.createSuperpixelSEEDS(
            lab_img.shape[1],
            lab_img.shape[0],
            lab_img.shape[2],
            num_superpixels=pbn_config.SEED_NUM_SUPERPIXELS,
            num_levels=pbn_config.SEED_NUM_LEVELS,
            prior=pbn_config.SEED_PRIOR,
            histogram_bins=pbn_config.SEED_HISTOGRAM_BINS,
            double_step=True,
        )
        sp.iterate(lab_img, pbn_config.SEED_NUM_ITERATIONS)
    else:
        raise ValueError("指定了错误的超像素算法，请选择 SLIC 或 SEED！")
    # 4. 获取超像素标签和数量
    splabels = sp.getLabels()  # 获取超像素标签 (1 ~ scout)
    spcount = sp.getNumberOfSuperpixels()  # 获取超像素数目
    # 5. 画出超像素分割后的图
    mask = sp.getLabelContourMask()
    mask_inv_seeds = cv2.bitwise_not(mask)
    sp_img = cv2.bitwise_and(img, img, mask=mask_inv_seeds)
    # 6. 生成 spuerpixel 组成的特征图
    feature_list = []
    # 每一个 superpixel 的颜色取其包含的所有原像素的均值
    for slbl in tqdm(range(1, spcount + 1), desc="生成超像素特征图"):
        mask = splabels == slbl
        mask = mask.astype(np.uint8)
        # 颜色取均值
        mean_color = cv2.mean(img, mask=mask)[:3]
        feature_list.append([mean_color[0], mean_color[1], mean_color[2]])
    sp_features = np.array(feature_list, dtype=np.float32)

    # 再用超像素特征图进行 K-Means 聚类
    # 1. K-Means 算法
    criteria = (
        pbn_config.KMEANS_CRITERIA_TYPE,
        pbn_config.KMEANS_CRITERIA_MAX_ITER,
        pbn_config.KMEANS_CRITERIA_EPSILON,
    )
    logger.info(
        f"正在进行 K-Means 计算（K - {pbn_config.KMEANS_NCLUSTERS}, "
        f"Attempts - {pbn_config.KMEANS_ATTEMPTS}, "
        f"MaxIter - {pbn_config.KMEANS_CRITERIA_MAX_ITER}, "
        f"Epsilon - {pbn_config.KMEANS_CRITERIA_EPSILON}）"
    )
    _, klabels, kcenters = cv2.kmeans(
        sp_features,
        pbn_config.KMEANS_NCLUSTERS,
        None,
        criteria,
        pbn_config.KMEANS_ATTEMPTS,
        pbn_config.KMEANS_FLAGS,
    )
    # 2. 生成聚类结果图
    clustered_img = np.zeros_like(img).reshape(-1, 3)
    splabels = splabels.flatten()
    klabels = klabels.flatten()
    for sp_lbl in tqdm(range(1, spcount + 1), desc="生成聚类结果图"):
        clustered_img[splabels == sp_lbl] = kcenters[klabels[sp_lbl - 1]].astype(int)
    clustered_img = clustered_img.reshape(img.shape)
    # 3. 生成各个区域的二值图
    area_parts = []
    for klbl in tqdm(range(kcenters.shape[0]), desc="生成各个区域的二值图"):
        # 每次挑出所有 klbl 标签的超像素，再找出这些超像素对应的所有原图像素，对应位置设为白色
        part = np.zeros(img.shape[:2], np.uint8).reshape((-1, 1))
        for spidx in range(klabels.size):
            if klabels[spidx] == klbl:
                # 超像素标签就是当前像素索引+1，因为标签从 1 开始，索引从 0 开始
                slbl = spidx + 1
                part[splabels == slbl] = 255
        part = part.reshape(img.shape[:2])
        area_parts.append(part)

    return sp_img, clustered_img, area_parts, kcenters


def _clusterize(img: MatLike) -> tuple[MatLike, list[MatLike], MatLike]:
    """K-Means 算法按区域分割图像

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
    logger.info(
        f"正在进行 K-Means 计算（K - {pbn_config.KMEANS_NCLUSTERS}, "
        f"Attempts - {pbn_config.KMEANS_ATTEMPTS}）"
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
    for label in tqdm(range(centers.shape[0]), desc="生成各个区域的二值图"):
        # part 是每个部分的图像，不需要通道维度，一开始是全黑，先展成一维以便计算
        part = np.zeros(img.shape[:2], np.uint8).reshape((-1, 1))
        for pixel_idx in range(part.size):
            # 如果该位置所属标签和当前标签相同，则该位置的像素设为白色（255)
            if labels[pixel_idx] == label:
                part[pixel_idx] = 255
        part = part.reshape(img.shape[:2])
        area_parts.append(part)

    return None, recolored_img, area_parts, centers


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
    # 白底轮廓图，颜色用 200 灰色模仿手绘风格
    contour_img = np.ones(img_shape, dtype=np.uint8) * 255
    for part in tqdm(area_parts, desc="正在绘制 PBN 图像："):
        # findContours 会找黑底图中的白色对象
        contours, hierarchy = cv2.findContours(
            part, pbn_config.CONTOUR_RETRIEVAL_MODE, pbn_config.CONTOUR_APPROX_MODE
        )
        # 过滤层级
        # filtered_contour_idx = [0]
        # info = hierarchy[0][0]
        # while True:
        #     next_idx = info[0]
        #     if next_idx == -1:
        #         break
        #     filtered_contour_idx.append(next_idx)
        #     info = hierarchy[0][next_idx]

        # rest_contours = []
        # for idx in filtered_contour_idx:
        #     rest_contours.append(contours[idx])

        # 过滤面积小于 min_area 的轮廓区域
        filtered_contours = [
            cntr for cntr in contours if cv2.contourArea(cntr) > pbn_config.MIN_AREA
        ]
        # approx_contours = []
        # for cnt in filtered_contours:
        #     epsilon = 0.005 * cv2.arcLength(cnt, True)
        #     approx_contours.append(cv2.approxPolyDP(cnt, epsilon, True))

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

        cv2.drawContours(
            image=contour_img,
            contours=filtered_contours,
            contourIdx=-1,
            color=pbn_config.CONTOUR_COLOR,
            thickness=pbn_config.CONTOUR_LINE_THICKNESS,
            lineType=8,
            hierarchy=hierarchy,
            maxLevel=2,
        )

    # 绘制图像底部的颜色展示面板
    bott_panel = np.zeros((50, contour_img.shape[1], 3), dtype=np.uint8)
    average_width = bott_panel.shape[1] // centers.shape[0]
    rect_width = average_width
    # 为每一个颜色创建矩形
    for center_idx in range(centers.shape[0]):
        center_color = centers[center_idx]  # 中心颜色
        # 色块
        cv2.rectangle(
            img=bott_panel,
            pt1=(center_idx * average_width, 48),
            pt2=(rect_width - 1 + center_idx * average_width, 2),
            color=[int(col) for col in center_color],
            thickness=-1,
        )
        # 外圈
        cv2.rectangle(
            img=bott_panel,
            pt1=(center_idx * average_width, 49),
            pt2=(rect_width + center_idx * average_width, 1),
            color=(255, 255, 255),
        )
        # 文字
        cv2.putText(
            img=bott_panel,
            text=f"{center_idx + 1}",
            org=(center_idx * average_width, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1.2,
            color=(255, 255, 255),
            thickness=2,
        )

    # 将轮廓二值图（当成灰度图）  转换为 BGR 图像
    pbn_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
    if pbn_config.SHOW_BOTTOM_PANEL:
        # 拼接轮廓图和颜色面板
        pbn_img = np.vstack((pbn_img, bott_panel))

    return pbn_img
