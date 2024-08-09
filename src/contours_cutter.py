import os
import cv2

from cv2.typing import MatLike
import numpy as np
from xu_pytools.configs import pbn_config, app_config


def _canny(img):
    canny_img = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
    canny_img = cv2.bitwise_not(canny_img)
    return canny_img


def cut_contours(img: MatLike):
    canny_img = _canny(img)
    saved_path = os.path.join(app_config.OUTPUT_DIR, "car_pbn_canny.jpg")
    cv2.imwrite(saved_path, canny_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        img_gray,
        mode=pbn_config.CONTOUR_RETRIEVAL_MODE,
        method=pbn_config.CONTOUR_APPROX_MODE,
    )

    single_contour_imgs = []
    for cntr_idx in range(len(contours)):
        scimg = np.ones(img.shape[:2], dtype=np.uint8) * 255
        cv2.drawContours(
            image=scimg,
            contours=contours,
            contourIdx=cntr_idx,
            color=pbn_config.CONTOUR_COLOR,
            thickness=pbn_config.CONTOUR_LINE_THICKNESS,
            lineType=8,
            hierarchy=hierarchy,
            maxLevel=0,
        )
        single_contour_imgs.append(scimg)

    return single_contour_imgs
