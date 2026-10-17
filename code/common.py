# ----------------  Спільні функції для lab2, lab3 і model_detection  ---------------

'''
Shared by code/lab2, code/lab3 and code/model_detection:
paths, image loading, numbered output folders (output/runX_labN), small morphology helpers
and drawing of the numbered building outlines.
'''

import os
import re
import cv2
import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.normpath(os.path.join(CODE, '..', 'photos'))
OUTPUT = os.path.normpath(os.path.join(CODE, '..', 'output'))
IMAGES = ['KPI_esri', 'KPI']


def image_read(file_im, file_roi=None):
    image = cv2.imread(file_im)
    if image is None:
        raise FileNotFoundError(file_im)
    roi = None
    if file_roi and os.path.exists(file_roi):
        roi = cv2.imread(file_roi, cv2.IMREAD_GRAYSCALE) > 0
    if roi is None:
        roi = np.ones(image.shape[:2], bool)
    return image, roi


def load(name):
    return image_read(os.path.join(PHOTOS, name + '.jpg'), os.path.join(PHOTOS, name + '_roi_mask.png'))


def fill_holes(mask):
    ff = mask.copy()
    h, w = mask.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 128)
    return np.where(ff == 128, 0, 255).astype(np.uint8)


def new_run_dir(lab):
    # output/run1_lab2, output/run2_lab2, ... - numbered separately for every lab
    os.makedirs(OUTPUT, exist_ok=True)
    pattern = re.compile(rf'^run(\d+)_{lab}$')
    runs = [int(m.group(1)) for d in os.listdir(OUTPUT) if (m := pattern.match(d))]
    run_dir = os.path.join(OUTPUT, f'run{max(runs, default=0) + 1}_{lab}')
    os.makedirs(run_dir)
    return run_dir


def ellipse(k):
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def label_point(contour, shape):
    # the point deepest inside the polygon (works for L/U-shaped buildings too)
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [contour], -1, 255, -1)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    y, x = np.unravel_index(dt.argmax(), dt.shape)
    return int(x), int(y)


def draw_buildings(image, polys, roi):
    out = image.copy()
    roi_cnt = cv2.findContours(roi.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(out, roi_cnt, -1, (0, 255, 0), 1)
    cv2.drawContours(out, polys, -1, (0, 0, 255), 2)
    for i, c in enumerate(polys, 1):
        x, y = label_point(c, image.shape[:2])
        (tw, th), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        org = (x - tw // 2, y + th // 2)
        cv2.putText(out, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(out, f'Buildings: {len(polys)}', (10, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(out, f'Buildings: {len(polys)}', (10, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
