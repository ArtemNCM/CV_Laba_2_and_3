# ----------------  Lab 2: порівняння одного й того ж об'єкта на різних зображеннях  ---------------

'''
Proof that the extracted features identify a building: the same building found on
KPI_esri (Landsat Explorer) and on KPI (Bing Maps) must be more similar than two different
buildings.

 1. Photo alignment  - SIFT key points of both photos, FLANN matching with the ratio test
                       (as in the lecture example gabor_filter.py -> sift_feature_matching),
                       homography by RANSAC; the reference annotation is NOT used
 2. Pairs            - buildings found on KPI are moved into KPI_esri coordinates and paired
                       with KPI_esri buildings by polygon overlap (IoU >= 0.3)
 3. Features         - shape: Hu moments of the contour, distance cv2.matchShapes (I1);
                       appearance: number of good SIFT matches between the building crops
 4. Test             - medians for the same building vs different buildings, and
                       identification: is the most similar building the right one?
'''

import os
import cv2
import numpy as np

from common import load

RATIO = 0.7          # Lowe's ratio test, as in the lecture example
MIN_IOU = 0.3        # polygons overlap needed to call two detections the same building
PAD = 8              # px around the building crop
UPSCALE = 3          # crops are small; enlarge them so SIFT finds enough key points


def sift_matches(img1, img2, sift):
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return k1, k2, []
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    pairs = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    return k1, k2, good


def photo_homography(img_a, img_b):
    # KPI_esri -> KPI, from the photos themselves
    sift = cv2.SIFT_create(5000)
    ga, gb = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    ka, kb, good = sift_matches(ga, gb, sift)
    p = np.float32([ka[m.queryIdx].pt for m in good])
    q = np.float32([kb[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(p, q, cv2.RANSAC, 3.0)
    return H, len(good), int(inliers.sum())


def poly_mask(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [poly], -1, 255, -1)
    return m > 0


def pair_buildings(polys_a, polys_b_in_a, shape):
    ma = [poly_mask(p, shape) for p in polys_a]
    mb = [poly_mask(p, shape) for p in polys_b_in_a]
    cand = []
    for i, a in enumerate(ma):
        for j, b in enumerate(mb):
            inter = (a & b).sum()
            if inter:
                cand.append((inter / (a | b).sum(), i, j))
    pairs, used_a, used_b = [], set(), set()
    for iou, i, j in sorted(cand, reverse=True):
        if iou >= MIN_IOU and i not in used_a and j not in used_b:
            pairs.append((i, j, iou))
            used_a.add(i), used_b.add(j)
    return pairs


def crop(gray, poly, scale=1.0):
    x, y, w, h = cv2.boundingRect(poly)
    x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
    c = gray[y0:y + h + PAD, x0:x + w + PAD]
    return cv2.resize(c, None, fx=UPSCALE * scale, fy=UPSCALE * scale, interpolation=cv2.INTER_CUBIC)


def compare(polys_esri, polys_kpi, out_dir=None):
    img_e, _ = load('KPI_esri')
    img_k, _ = load('KPI')
    H, n_good, n_inl = photo_homography(img_e, img_k)
    Hinv = np.linalg.inv(H)
    kpi_in_esri = [np.int32(np.round(cv2.perspectiveTransform(p.astype(np.float64), Hinv))) for p in polys_kpi]
    pairs = pair_buildings(polys_esri, kpi_in_esri, img_e.shape[:2])
    lines = [f'Same building on both photos (SIFT homography of the photos: {n_good} matches, {n_inl} inliers)',
             f'  {len(pairs)} buildings paired by polygon overlap (IoU >= {MIN_IOU})']
    if len(pairs) < 2:
        return lines + ['  not enough pairs for the comparison']

    # scale between the photos (KPI -> KPI_esri) so the crops have the same size
    scale = float(np.sqrt(abs(np.linalg.det(Hinv[:2, :2]))))
    sift = cv2.SIFT_create()
    ge, gk = cv2.cvtColor(img_e, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_k, cv2.COLOR_BGR2GRAY)
    crops_e = [crop(ge, polys_esri[i]) for i, _, _ in pairs]
    crops_k = [crop(gk, polys_kpi[j], scale) for _, j, _ in pairs]

    n = len(pairs)
    hu = np.zeros((n, n))
    sm = np.zeros((n, n))
    for a, (i, _, _) in enumerate(pairs):
        for b, (_, j, _) in enumerate(pairs):
            hu[a, b] = cv2.matchShapes(polys_esri[i], kpi_in_esri[j], cv2.CONTOURS_MATCH_I1, 0)
            sm[a, b] = len(sift_matches(crops_e[a], crops_k[b], sift)[2])
    same = np.eye(n, dtype=bool)
    id_hu = int((hu.argmin(axis=1) == np.arange(n)).sum())
    id_sift = int((sm.argmax(axis=1) == np.arange(n)).sum())
    lines += ['  feature                        same building   different buildings   (median)',
              f'  shape: Hu moments distance I1   {np.median(hu[same]):12.3f}   {np.median(hu[~same]):16.3f}',
              f'  SIFT good matches of the crops  {np.median(sm[same]):12.1f}   {np.median(sm[~same]):16.1f}',
              '  identification (most similar building among the paired ones, positions not used):',
              f'    by shape (Hu moments)      {id_hu:3d}/{n}',
              f'    by SIFT matches            {id_sift:3d}/{n}',
              '  pairs: esri# kpi#  IoU  Hu-dist  SIFT-matches']
    for a, (i, j, iou) in enumerate(pairs):
        lines.append(f'    {i + 1:3d} {j + 1:4d}  {iou:.2f}  {hu[a, a]:7.3f}  {int(sm[a, a]):6d}')

    if out_dir:                                          # example: matches for the best pair
        a = int(sm.diagonal().argmax())
        k1, k2, good = sift_matches(crops_e[a], crops_k[a], sift)
        vis = cv2.drawMatches(crops_e[a], k1, crops_k[a], k2, good, None, matchColor=(0, 255, 0),
                              singlePointColor=(255, 0, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
        cv2.imwrite(os.path.join(out_dir, 'same_building_sift.png'), vis)
        b = (a + 1) % n                                  # and for two different buildings
        k1, k2, good = sift_matches(crops_e[a], crops_k[b], sift)
        vis = cv2.drawMatches(crops_e[a], k1, crops_k[b], k2, good, None, matchColor=(0, 255, 0),
                              singlePointColor=(255, 0, 0), flags=cv2.DrawMatchesFlags_DEFAULT)
        cv2.imwrite(os.path.join(out_dir, 'different_buildings_sift.png'), vis)
        lines.append(f'  figures: same_building_sift.png (pair {pairs[a][0] + 1}/{pairs[a][1] + 1}), '
                     f'different_buildings_sift.png')
    return lines
