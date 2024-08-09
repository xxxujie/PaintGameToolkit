import os
import cv2
import numpy as np
from src.contours_cutter import cut_contours
from src import pbn_generator
from xu_pytools import loggers, configs


def main():
    logger.info("Start running")
    # pbn_generator.img2pbn(configs.app_config.DATA_DIR)
    img_path = os.path.join(configs.app_config.DATA_DIR, "car_part1.jpg")
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        img_gray,
        mode=configs.pbn_config.CONTOUR_RETRIEVAL_MODE,
        method=configs.pbn_config.CONTOUR_APPROX_MODE,
    )

    single_contour_imgs = []
    for cntr_idx in range(len(contours)):
        scimg = np.ones(img.shape[:2], dtype=np.uint8) * 255
        cv2.drawContours(
            image=scimg,
            contours=contours,
            contourIdx=cntr_idx,
            color=configs.pbn_config.CONTOUR_COLOR,
            thickness=configs.pbn_config.CONTOUR_LINE_THICKNESS,
            lineType=8,
            hierarchy=hierarchy,
            maxLevel=0,
        )
        single_contour_imgs.append(scimg)

    idx = 0
    for cnimg in single_contour_imgs:
        if not os.path.exists(configs.app_config.OUTPUT_DIR):
            os.makedirs(configs.app_config.OUTPUT_DIR)
        saved_path = os.path.join(
            configs.app_config.OUTPUT_DIR, f"car_part1_cntr{idx}.jpg"
        )
        cv2.imwrite(saved_path, cnimg)
        idx += 1


if __name__ == "__main__":
    logger = loggers.get_logger()
    main()
