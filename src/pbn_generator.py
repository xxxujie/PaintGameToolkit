import json
import os
import cv2
import numpy as np

from tqdm import tqdm
from cv2.typing import MatLike
from xu_pytools import loggers
from xu_pytools.configs import pbn_config, app_config


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


def _save_img(img: MatLike, img_name, tag="", ext=None):
    name_without_ext = os.path.splitext(img_name)[0]
    name_ext = os.path.splitext(img_name)[1]
    if ext is not None:
        name_ext = ext
    if not os.path.exists(app_config.OUTPUT_DIR):
        os.makedirs(app_config.OUTPUT_DIR)
    saved_path = os.path.join(app_config.OUTPUT_DIR, name_without_ext + tag + name_ext)
    cv2.imwrite(saved_path, img)
    return saved_path


def _convert_and_save(img_path: str):
    img = cv2.imread(img_path)
    img_name = os.path.basename(img_path)
    logger.info(f"开始转换 PBN（for {img_name}）")
    # 方法一：先通过聚类分离原图区域，再画出轮廓图
    # slic_img, recolored_img, area_parts, centers = _clusterize(img)
    superpixel_img, recolored_img, area_parts, centers = _cluster_with_superpixel(
        img, pbn_config.SUPERPIXEL_ALGORITHM
    )
    only_cntr_img, info4area, single_contour_imgs, area_gray_imgs, pbn_img = (
        _draw_outline(img, area_parts, centers)
    )

    # 方法二：Canny 边缘检测算法分割
    # canny_img = _canny(img)
    # _save_img(canny_img, img_name, "_canny")

    # 方法三：自定义阈值分割
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten()
    # custom_img = np.ones(img_gray.shape, dtype=np.uint8) * 255
    # for idx in range(img_gray.size):
    #     if img_gray[idx] < 100:
    #         custom_img[idx] = 0
    # custom_img = custom_img.reshape(img.shape[:2])
    # custom_img = cv2.cvtColor(custom_img, cv2.COLOR_GRAY2BGR)
    # _save_img(custom_img, img_name, "_custom")

    # 保存所有单个的白色区域图为 png，因为需要透明效果
    # for idx in range(len(single_contour_imgs)):
    #     _save_img(single_contour_imgs[idx], img_name, f"_part_{idx}", ".png")
    # 保存所有单个的灰度区域图为 png
    for idx in range(len(area_gray_imgs)):
        _save_img(area_gray_imgs[idx], img_name, f"_part_{idx}", ".png")
    # 保存各部分的色块图
    # _save_parts(area_parts, img_name)
    # 保存超像素结果
    # _save_img(superpixel_img, img_name, "_superpixel")
    # 保存重上色（聚类）结果
    # _save_img(recolored_img, img_name, "_recolored")
    # 保存仅线稿图（只有轮廓，其他透明）
    _save_img(only_cntr_img, img_name, "_contours", ".png")
    # 保存完整线稿图
    _save_img(pbn_img, img_name, "_pbn", ".png")
    # 保存颜色和其索引为 JSON 文件
    img_name_prefix = os.path.splitext(img_name)[0]
    color_path = os.path.join(app_config.OUTPUT_DIR, f"{img_name_prefix}_colors.json")
    info4color = {"info": []}
    for color_idx, color in enumerate(centers):
        info4color["info"].append({"idx": color_idx, "color": [int(c) for c in color]})
    with open(color_path, "w", encoding="utf-8") as f:
        json.dump(info4color, f)
    # 保存中心点坐标为 JSON 文件
    data_path = os.path.join(app_config.OUTPUT_DIR, f"{img_name_prefix}_info.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(info4area, f)

    logger.info(f"转换完成！（Results are saved at {app_config.OUTPUT_DIR}）")


def _save_parts(parts, img_name):
    """保存每个部分的色块图"""

    cnt = 0
    for part in parts:
        _save_img(part, img_name, f"_part{cnt}")
        cnt += 1
    return cnt


def _canny(img):
    """Canny 边缘检测算法"""

    canny_img = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    canny_img = cv2.bitwise_not(canny_img)
    return canny_img


def _cluster_with_superpixel(img: MatLike, sp_algorithm):
    """先进行超像素分割，再用 K-Means 聚类

    Args:
        img (MatLike): 输入图片
        sp_algorithm (str): 超像素算法（SLIC 或 SEED）
    """
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
    splabels = sp.getLabels()  # 获取超像素标签 (0 ~ num_superpixels)
    spcount = sp.getNumberOfSuperpixels()  # 获取超像素数目
    # 5. 画出超像素分割后的图
    mask = sp.getLabelContourMask()
    mask_inv_seeds = cv2.bitwise_not(mask)
    sp_img = cv2.bitwise_and(img, img, mask=mask_inv_seeds)
    # 6. 生成 spuerpixel 组成的特征图
    feature_list = []
    # 每一个 superpixel 的颜色取其包含的所有原像素的均值
    for slbl in tqdm(range(0, spcount), desc="生成超像素特征图"):
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
                part[splabels == spidx] = 255
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


def _draw_outline(img: MatLike, area_parts: list[MatLike], centers: MatLike):
    """根据多个区域二值图和簇心画出轮廓图

    Args:
        img (MatLike): 原图
        area_parts (list[MatLike]): 各标签对应的区域图像数组
        centers (MatLike): K-Means 获得的簇心集合

    Returns:
        MatLike: 画出轮廓后的图像
    """
    # 二值图像的尺寸以原图为基准
    img_shape = img.shape[:2]
    # 灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_back_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    gb, gg, gr = cv2.split(gray_img_back_bgr)
    # 一整张轮廓图，白底开始画
    contour_img = np.ones(img_shape, dtype=np.uint8) * 255
    # 每个区域分别的白色区域图
    single_contour_imgs = []
    # 每个区域分析的灰度区域图
    area_gray_imgs = []
    info4area = {"info": []}
    contour_idx = 0
    logger.info("正在绘制 PBN 图像...")
    for ap_idx, areapart in enumerate(area_parts):
        # findContours 会找黑底图中的白色对象
        contours, hierarchy = cv2.findContours(
            areapart, pbn_config.CONTOUR_RETRIEVAL_MODE, pbn_config.CONTOUR_APPROX_MODE
        )

        # 过滤面积小于 min_area 的轮廓区域
        filtered_contours = [
            cntr for cntr in contours if cv2.contourArea(cntr) > pbn_config.MIN_AREA
        ]
        # 计算每个轮廓的中心点，并且每个轮廓画一张单张轮廓图
        for cntr in filtered_contours:
            # 在一张黑底图上画单个白色区域
            single_cntr_img = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(
                image=single_cntr_img,
                contours=[cntr],
                contourIdx=-1,
                color=[255, 255, 255],
                thickness=-1,  # 用填充的方式
                lineType=8,
            )
            # 黑色背景变透明
            _, alpha = cv2.threshold(single_cntr_img, 1, 255, cv2.THRESH_BINARY)
            single_cntr_img_bgr = cv2.cvtColor(single_cntr_img, cv2.COLOR_GRAY2BGR)
            b, g, r = cv2.split(single_cntr_img_bgr)
            dst_scimg = cv2.merge((b, g, r, alpha))
            # 拿到灰度版本的区域（通过位与，黑色区域还会保持黑色）
            gray_img_with_alpha = cv2.merge((gb, gg, gr, alpha))
            gray_dst_scimg = cv2.bitwise_and(gray_img_with_alpha, dst_scimg)
            # 找到该轮廓的最小外接矩形，裁剪这个矩形出来
            leftmost = tuple(cntr[cntr[:, :, 0].argmin()][0])
            rightmost = tuple(cntr[cntr[:, :, 0].argmax()][0])
            topmost = tuple(cntr[cntr[:, :, 1].argmin()][0])
            bottommost = tuple(cntr[cntr[:, :, 1].argmax()][0])
            x_min = leftmost[0]
            x_max = rightmost[0]
            y_min = topmost[1]
            y_max = bottommost[1]
            roi = dst_scimg[y_min:y_max, x_min:x_max]
            gray_roi = gray_dst_scimg[y_min:y_max, x_min:x_max]
            # 计算每个区域图的中心点坐标，并记录
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            area_pos = (int(center_x), int(center_y))

            # Optional：近似轮廓
            # approx_contours = []
            # for cnt in filtered_contours:
            #     epsilon = 0.005 * cv2.arcLength(cnt, True)
            #     approx_contours.append(cv2.approxPolyDP(cnt, epsilon, True))

            # Optional：在轮廓中标号，并记录
            # 先找到轮廓中心，用于标号位置
            # M = cv2.moments(cntr)
            # cx = int(M["m10"] / M["m00"])
            # cy = int(M["m01"] / M["m00"])
            # offset = 5  # 标号位置偏移量
            # cx -= x_min + offset
            # cy -= y_min - offset
            # if pbn_config.SHOW_AREA_INDEX:
            #     cv2.putText(  # 在区域内写上标号
            #         img=roi,
            #         text=str(ap_idx + 1),  # 写在图上的标号从 1 开始
            #         org=(int(cx), int(cy)),
            #         fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #         fontScale=0.5,
            #         color=pbn_config.CONTOUR_COLOR,
            #     )

            single_contour_imgs.append(roi)
            area_gray_imgs.append(gray_roi)

            # 该区域颜色索引
            area_color = ap_idx

            # 将区域的信息记录到字典中
            info4area["info"].append(
                {
                    "idx": contour_idx,
                    "color_idx": area_color,
                    "pos": area_pos,
                }
            )
            contour_idx += 1

        # 画整张轮廓图
        cv2.drawContours(
            image=contour_img,
            contours=filtered_contours,
            contourIdx=-1,
            color=pbn_config.CONTOUR_COLOR,
            thickness=pbn_config.CONTOUR_LINE_THICKNESS,
            lineType=8,
            # hierarchy=hierarchy,
            # maxLevel=2,
        )

    # 将整张轮廓图的白色背景变透明
    # 1. 生成与白色部分对应的mask图像
    pbn_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
    mask = np.all(pbn_img[:, :, :] == [255, 255, 255], axis=-1)
    # 2. 将图片从三通道转为四通道
    only_cntr_img = cv2.cvtColor(pbn_img, cv2.COLOR_BGR2BGRA)
    # 3. 以mask图像为基础，使白色部分透明化
    only_cntr_img[mask, 3] = 0

    return only_cntr_img, info4area, single_contour_imgs, area_gray_imgs, pbn_img
