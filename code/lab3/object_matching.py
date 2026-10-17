# ----------------  Lab 3: порівняння одного й того ж об'єкта на різних зображеннях  ---------------

'''
Proof that the extracted features are suitable for identification:
the same building found on KPI_esri and on KPI must have similar features,
different buildings must have different ones.

 1. Both photos are aligned with a homography (SIFT + RANSAC, evaluate_reference.py);
    buildings detected on KPI are moved into KPI_esri coordinates.
 2. Buildings are paired by overlap (IoU of their polygons).
 3. Features of every building:
      shape  - area, rectangularity (area / minAreaRect), elongation, solidity,
               Hu moments -> cv2.matchShapes distance
      colour - mean Lab colour of the roof
 4. Feature distances are compared for the SAME building (matched pairs) and for
    DIFFERENT buildings (each building against all other buildings).
    The colour distance is computed on the raw photos and after colour correction.
'''

import cv2
import numpy as np


def warp_polys(polys, H):
    return [np.int32(np.round(cv2.perspectiveTransform(p.astype(np.float64), H))) for p in polys]


def poly_mask(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [poly], -1, 255, -1)
    return m > 0


def match_buildings(polys_a, polys_b, shape, min_iou=0.3):
    # greedy pairing by polygon IoU (both lists in the same coordinates)
    ma = [poly_mask(p, shape) for p in polys_a]
    mb = [poly_mask(p, shape) for p in polys_b]
    cand = []
    for i, a in enumerate(ma):
        for j, b in enumerate(mb):
            inter = (a & b).sum()
            if inter:
                cand.append((inter / (a | b).sum(), i, j))
    pairs, used_a, used_b = [], set(), set()
    for iou, i, j in sorted(cand, reverse=True):
        if iou >= min_iou and i not in used_a and j not in used_b:
            pairs.append((i, j, iou))
            used_a.add(i), used_b.add(j)
    return pairs


def shape_features(poly):
    area = cv2.contourArea(poly)
    (_, _), (w, h), _ = cv2.minAreaRect(poly)
    hull = cv2.contourArea(cv2.convexHull(poly))
    return dict(area=area, rect=area / max(w * h, 1), elong=max(w, h) / max(min(w, h), 1),
                solidity=area / max(hull, 1))


def mean_lab(image, poly):
    m = poly_mask(poly, image.shape[:2])
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0     # roof only, no border
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[m].astype(np.float64).mean(axis=0)


def delta_e(lab1, lab2):
    # OpenCV 8-bit Lab: L in 0..255, a/b shifted by 128 -> back to CIE scale
    k = np.array([100 / 255, 1, 1])
    return float(np.linalg.norm((lab1 - lab2) * k))


def compare(polys_esri, polys_kpi, img_esri, img_kpi, H_esri_to_kpi, colour_versions):
    '''
    colour_versions: {'raw': (img_esri, img_kpi), 'gray_world': (...), ...}
    Returns rows for matched pairs and a summary: same building vs different buildings.
    '''
    shape = img_esri.shape[:2]
    kpi_in_esri = warp_polys(polys_kpi, np.linalg.inv(H_esri_to_kpi))
    pairs = match_buildings(polys_esri, kpi_in_esri, shape)

    rows = []
    for i, j, iou in pairs:
        fa, fb = shape_features(polys_esri[i]), shape_features(kpi_in_esri[j])
        row = dict(esri=i + 1, kpi=j + 1, iou=iou,
                   area_ratio=fb['area'] / max(fa['area'], 1),
                   d_rect=abs(fa['rect'] - fb['rect']), d_solidity=abs(fa['solidity'] - fb['solidity']),
                   hu=cv2.matchShapes(polys_esri[i], kpi_in_esri[j], cv2.CONTOURS_MATCH_I1, 0))
        for name, (ie, ik) in colour_versions.items():
            row['dE_' + name] = delta_e(mean_lab(ie, polys_esri[i]), mean_lab(ik, polys_kpi[j]))
        rows.append(row)

    # different buildings: every matched esri building vs every OTHER matched kpi building
    other = dict(hu=[], **{'dE_' + n: [] for n in colour_versions})
    for i, _, _ in pairs:
        for _, j, _ in pairs:
            if (i, j) in {(a, b) for a, b, _ in pairs}:
                continue
            other['hu'].append(cv2.matchShapes(polys_esri[i], kpi_in_esri[j], cv2.CONTOURS_MATCH_I1, 0))
            for name, (ie, ik) in colour_versions.items():
                other['dE_' + name].append(delta_e(mean_lab(ie, polys_esri[i]), mean_lab(ik, polys_kpi[j])))

    summary = {k: (float(np.median([r[k] for r in rows])) if rows else float('nan'),
                   float(np.median(v)) if v else float('nan')) for k, v in other.items()}

    # identification test: for every building on KPI_esri, is its true partner on KPI
    # the nearest one in feature space?  (positions are NOT used)
    def descriptor(poly, img, poly_own):
        f = shape_features(poly)
        lab = mean_lab(img, poly_own)
        return dict(shape=[np.log(f['area']), f['rect'], np.log(f['elong']), f['solidity']],
                    colour=list(lab))

    col = colour_versions.get('gray_world', next(iter(colour_versions.values())))
    da = [descriptor(polys_esri[i], col[0], polys_esri[i]) for i, _, _ in pairs]
    db = [descriptor(kpi_in_esri[j], col[1], polys_kpi[j]) for _, j, _ in pairs]
    ident = {}
    for kind, keys in [('shape', ['shape']), ('colour', ['colour']), ('shape + colour', ['shape', 'colour'])]:
        A = np.array([sum((d[k] for k in keys), []) for d in da])
        B = np.array([sum((d[k] for k in keys), []) for d in db])
        A, B = A - A.mean(axis=0), B - B.mean(axis=0)                 # remove each photo's offset
        sd = np.vstack([A, B]).std(axis=0) + 1e-9                     # standardize every feature
        D = np.linalg.norm((A[:, None, :] - B[None, :, :]) / sd, axis=2)
        ident[kind] = int((D.argmin(axis=1) == np.arange(len(pairs))).sum())
    ident['Hu moments'] = sum(
        int(np.argmin([cv2.matchShapes(polys_esri[i], kpi_in_esri[jj], cv2.CONTOURS_MATCH_I1, 0)
                       for _, jj, _ in pairs]) == k) for k, (i, _, _) in enumerate(pairs))
    return rows, summary, ident, len(pairs)


def report_lines(rows, summary, ident, n):
    lines = [f'Same building on both photos: {n} matched pairs (polygon IoU >= 0.3)',
             '  distance                same building   different buildings   (median)']
    for k, (same, diff) in summary.items():
        lines.append(f'  {k:22s}  {same:12.3f}   {diff:16.3f}')
    lines.append('  identification (nearest building in feature space, positions not used):')
    for kind, correct in ident.items():
        lines.append(f'    {kind:16s} {correct:3d}/{n} buildings matched to the right partner')
    lines.append('  pairs: esri# kpi#  IoU  area_ratio  Hu-dist  ' +
                 '  '.join(k for k in rows[0] if k.startswith('dE_')) if rows else '  no pairs')
    for r in rows:
        lines.append(f"    {r['esri']:3d} {r['kpi']:4d}  {r['iou']:.2f}  {r['area_ratio']:6.2f}  {r['hu']:7.3f}  " +
                     '  '.join(f'{r[k]:6.1f}' for k in r if k.startswith('dE_')))
    return lines
