# ----------------  Перевірка результату за еталонною розміткою KPI_esri_annotated_reference  ---------------

'''
The reference (KPI_esri_annotated_reference.png) has red outlines of 31 buildings, but it is
rotated/scaled relative to the photos. Steps:

 1. red outlines are extracted from the reference and filled -> building mask
 2. reference -> KPI_esri homography is found with SIFT features + RANSAC
    (KPI_esri -> KPI too, so the same reference is used for the older KPI photo)
 3. the warped reference mask is compared with the detected mask:
    pixel precision / recall / IoU and how many reference buildings were found

Shared by lab2 and lab3: buildings_detection.py of each lab calls it after detection.
'''

import os
import cv2
import numpy as np

from common import PHOTOS, fill_holes

REF = os.path.join(PHOTOS, 'KPI_esri_annotated_reference.png')


def reference_mask():
    ref = cv2.imread(REF)
    b, g, r = [ref[..., i].astype(int) for i in range(3)]
    red = ((r > 200) & (g < 60) & (b < 60)).astype(np.uint8) * 255
    clean = cv2.inpaint(ref, cv2.dilate(red, np.ones((5, 5))), 3, cv2.INPAINT_TELEA)
    outline = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9)))
    mask = cv2.morphologyEx(fill_holes(outline), cv2.MORPH_OPEN, np.ones((5, 5)))
    return clean, mask


def homography(src, dst):
    sift = cv2.SIFT_create(5000)
    ks, ds = sift.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    kd, dd = sift.detectAndCompute(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(ds, dd, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]       # Lowe ratio test
    p = np.float32([ks[m.queryIdx].pt for m in good])
    q = np.float32([kd[m.trainIdx].pt for m in good])
    H, _ = cv2.findHomography(p, q, cv2.RANSAC, 3.0)
    return H


def evaluate(name, det_mask, gt_mask, roi):
    det = (det_mask > 0) & roi
    gt = (gt_mask > 0) & roi
    tp = (det & gt).sum()
    prec, rec = tp / max(det.sum(), 1), tp / max(gt.sum(), 1)
    iou = tp / max((det | gt).sum(), 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8))
    blocks = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 150]   # skip label/text remnants
    found = sum(((lab == i) & det).sum() / st[i, cv2.CC_STAT_AREA] > 0.3 for i in blocks)
    nd, dl = cv2.connectedComponents(det.astype(np.uint8))
    false = sum(((dl == i) & gt).sum() / (dl == i).sum() < 0.2 for i in range(1, nd))
    return (f'vs reference: precision={prec:.2f} recall={rec:.2f} IoU={iou:.2f} | '
            f'reference blocks found {found}/{len(blocks)} | detections {nd - 1}, false {false}')


def overlay(image, det_mask, gt_mask):
    out = image.copy()
    gt_c = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    det_c = cv2.findContours(det_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(out, gt_c, -1, (255, 255, 0), 2)     # reference - cyan
    cv2.drawContours(out, det_c, -1, (0, 0, 255), 2)      # detected  - red
    return out


class Reference:
    # reference mask registered onto each photo (homographies are computed once)
    def __init__(self):
        clean_ref, self.mask = reference_mask()
        esri = cv2.imread(os.path.join(PHOTOS, 'KPI_esri.jpg'))
        kpi = cv2.imread(os.path.join(PHOTOS, 'KPI.jpg'))
        H_esri = homography(clean_ref, esri)
        self.H = {'KPI_esri': H_esri, 'KPI': homography(esri, kpi) @ H_esri}

    def compare(self, name, image, roi, det_mask, out_dir):
        if name not in self.H:
            return 'vs reference: no reference for this image'
        gt = cv2.warpPerspective(self.mask, self.H[name], image.shape[1::-1], flags=cv2.INTER_NEAREST)
        if out_dir:
            cv2.imwrite(os.path.join(out_dir, f'{name}_vs_reference.png'), overlay(image, det_mask, gt))
        return evaluate(name, det_mask, gt, roi)

