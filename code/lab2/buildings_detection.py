# ----------------  Lab 2: виявлення будівель - растр -> вектор (KPI, KPI_esri)  ---------------

'''
Simplified algorithm: colour transformations + Canny + contours (no image enhancement,
the photo is processed as it is).

 1. Colour transforms  - BGR -> HSV and Lab; roofs are grey/bluish
                         (low saturation S, low Lab-b), roads/paths are warmer
                         (higher Lab-b), vegetation is saturated green
 2. Thresholds         - roof mask from S, Lab-b and brightness L (percentiles of the
                         region of interest, so they adapt to each photo)
 3. Canny              - edges of the brightness channel; they cut the roof mask
                         where a roof touches a road or another roof
 4. Morphology         - opening removes thin roads, closing fills small gaps
 5. findContours       - raster mask -> vector contours
 6. Geometric filters  - area, width (roads are narrow), rectangularity
                         (area / minAreaRect area), solidity, mean colour
 7. Vectorization      - minAreaRect for rectangular objects, approxPolyDP for the rest;
                         numbering and counting

Package            Version
------------------ -----------
opencv-python      >= 4.8
numpy              >= 1.25
matplotlib         >= 3.7

Run:
    python buildings_detection.py             # process both photos, show results,
                                              # save everything to output/runX_lab2
    python buildings_detection.py --no-show   # only save, no windows
'''

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import (IMAGES, load, fill_holes, new_run_dir,  # noqa: E402
                    ellipse, draw_buildings)

LAB = 'lab2'

# Parameters are tuned for KPI_esri (800 px wide); pixel sizes are scaled by image width.
PARAMS = dict(
    base_width=800,
    roi_erode=11,          # px, cut the road that runs along the ROI border
    s_max=75,              # roof: HSV saturation below this
    b_pct=50,              # roof: Lab-b below this percentile of the ROI (grey-blue, not yellow)
    l_pct=30,              # roof: Lab-L above this percentile (not shadow)
    canny_low=150,         # Canny thresholds
    canny_high=375,
    edge_width=1,          # px, Canny edges are thickened to this width before cutting
    open_size=5,           # px, opening removes roads narrower than this
    close_size=9,          # px, closing fills small gaps in roofs
    min_area=200,          # px^2
    min_width=9,           # px, short side of minAreaRect (roads are narrower)
    min_rect=0.35,         # area / minAreaRect area
    min_solidity=0.45,     # area / convex hull area
    obj_b_pct=65,          # mean Lab-b of the object below this percentile
    rect_fill=0.75,        # object fills >= 75% of its rotated rectangle -> draw the rectangle
    poly_eps=0.015,        # approxPolyDP epsilon (fraction of perimeter)
)


def scaled(prm, width):
    # convert pixel parameters to the actual image resolution
    s = width / prm['base_width']
    p = dict(prm)
    for k in ['roi_erode', 'edge_width', 'open_size', 'close_size']:
        p[k] = max(1, int(round(prm[k] * s)))
    p['min_width'] = prm['min_width'] * s
    p['min_area'] = prm['min_area'] * s * s
    return p


# --------------------------- 1-2. colour transforms + thresholds -------------
def colour_mask(image, roi, p):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int32)
    H, S = hsv[..., 0], hsv[..., 1]
    L, B = lab[..., 0], lab[..., 2]

    bright = L > np.percentile(L[roi], p['l_pct'])
    grey_roof = (S < p['s_max']) & (B < np.percentile(B[roi], p['b_pct'])) & bright
    # green (copper-patina) roofs of the main building: cyan-green hue, moderate saturation
    green_roof = (H >= 60) & (H <= 95) & (S > 50) & (S < 160) & bright

    roi_in = cv2.erode(roi.astype(np.uint8) * 255, ellipse(p['roi_erode'])) > 0
    mask = ((grey_roof | green_roof) & roi_in).astype(np.uint8) * 255
    return mask, dict(L=L, B=B)


# --------------------------- 3. Canny edges ----------------------------------
def canny_edges(image, p):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)                            # as in Canny examples
    edges = cv2.Canny(gray, p['canny_low'], p['canny_high'])
    return cv2.dilate(edges, ellipse(p['edge_width']))


# --------------------------- 4. morphology -----------------------------------
def clean_mask(mask, edges, p):
    m = cv2.bitwise_and(mask, cv2.bitwise_not(edges))                  # cut along edges
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse(p['open_size']))    # remove thin roads
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse(p['close_size']))  # close gaps
    return fill_holes(m)


# --------------------------- 5-6. contours + geometric filtering -------------
def image_contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


def filter_contours(contours, feats, roi, p):
    t_b = np.percentile(feats['B'][roi], p['obj_b_pct'])
    kept, rejected = [], []                              # rejected: (contour, reason)
    for c in contours:
        area = cv2.contourArea(c)
        if area < p['min_area']:
            rejected.append((c, 'small'))
            continue
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        if min(w, h) < p['min_width']:                   # long and narrow -> road / path
            rejected.append((c, 'shape'))
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        if area / (w * h) < p['min_rect'] or area / max(hull, 1) < p['min_solidity']:
            rejected.append((c, 'shape'))                # ragged -> trees, road crossings
            continue
        m = np.zeros(roi.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        if feats['B'][m > 0].mean() > t_b:               # too warm -> road / sand / court
            rejected.append((c, 'colour'))
            continue
        kept.append(c)
    return kept, rejected


# --------------------------- 7. vectorization --------------------------------
def vectorize(contours, p):
    polys = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if cv2.contourArea(c) >= p['rect_fill'] * w * h:                 # rectangular building
            polys.append(np.int32(cv2.boxPoints(rect)).reshape(-1, 1, 2))
        else:                                                            # L / U / T shapes
            polys.append(cv2.approxPolyDP(c, p['poly_eps'] * cv2.arcLength(c, True), True))
    # number buildings top-to-bottom, left-to-right (rows ~ 40 px)
    polys.sort(key=lambda c: (int(cv2.boundingRect(c)[1] // 40), cv2.boundingRect(c)[0]))
    return polys


REJECT_COLOURS = {                                     # BGR
    'small': (0, 255, 255),        # yellow  - smaller than min_area
    'shape': (255, 0, 255),        # magenta - too narrow / ragged
    'colour': (255, 160, 0),       # blue    - too warm/yellow (road, sand, court)
}


def draw_debug(image, kept, rejected):
    # why was each contour kept or thrown away
    out = (image * 0.6).astype(np.uint8)
    layer = out.copy()
    for c, reason in rejected:
        cv2.drawContours(layer, [c], -1, REJECT_COLOURS[reason], -1)
    cv2.drawContours(layer, kept, -1, (0, 0, 255), -1)                  # accepted - red
    out = cv2.addWeighted(out, 0.45, layer, 0.55, 0)
    legend = [('accepted', (0, 0, 255)), ('rejected: too small', REJECT_COLOURS['small']),
              ('rejected: shape (narrow/ragged)', REJECT_COLOURS['shape']),
              ('rejected: colour (road/sand)', REJECT_COLOURS['colour'])]
    cv2.rectangle(out, (5, 5), (265, 12 + 18 * len(legend)), (0, 0, 0), -1)
    for i, (text, col) in enumerate(legend):
        y = 20 + 18 * i
        cv2.rectangle(out, (10, y - 9), (22, y + 3), col, -1)
        cv2.putText(out, text, (28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return out


# --------------------------- full pipeline ----------------------------------
def detect_buildings(image, roi, prm=PARAMS):
    p = scaled(prm, image.shape[1])
    mask, feats = colour_mask(image, roi, p)
    edges = canny_edges(image, p)
    clean = clean_mask(mask, edges, p)
    kept, rejected = filter_contours(image_contours(clean), feats, roi, p)
    polys = vectorize(kept, p)
    final = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(final, polys, -1, 255, -1)
    stages = dict(colour_mask=mask, canny=edges, clean_mask=clean, final_mask=final,
                  debug=draw_debug(image, kept, rejected))
    return polys, stages


def stages_figure(name, st, result):
    from matplotlib import pyplot as plt
    titles = ['Colour mask (HSV + Lab)', 'Canny edges', 'Mask cut by edges + morphology',
              'Kept / rejected contours', 'Buildings']
    imgs = [st['colour_mask'], st['canny'], st['clean_mask'], st['debug'], result]
    fig = plt.figure(figsize=(15, 9))
    for i, (t, im) in enumerate(zip(titles, imgs), 1):
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, 'gray')
        plt.title(t), plt.xticks([]), plt.yticks([])
    plt.suptitle(f'{name}: {len(st["polys"])} buildings')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def process(name, run_dir, log, reference=None):
    image, roi = load(name)
    polys, st = detect_buildings(image, roi)
    st['polys'] = polys
    result = draw_buildings(image, polys, roi)

    cv2.imwrite(os.path.join(run_dir, f'{name}_buildings.png'), result)
    cv2.imwrite(os.path.join(run_dir, f'{name}_mask.png'), st['final_mask'])
    cv2.imwrite(os.path.join(run_dir, f'{name}_debug_rejected.png'), st['debug'])
    for k in ['colour_mask', 'canny', 'clean_mask']:
        cv2.imwrite(os.path.join(run_dir, f'{name}_step_{k}.png'), st[k])
    fig = stages_figure(name, st, result)
    fig.savefig(os.path.join(run_dir, f'{name}_stages.png'), dpi=100)

    log(f'{name}: found {len(polys)} buildings')
    for i, c in enumerate(polys, 1):
        x, y, w, h = cv2.boundingRect(c)
        log(f'  #{i:2d}  center=({x + w // 2:3d},{y + h // 2:3d})  area={int(cv2.contourArea(c)):6d} px  vertices={len(c)}')

    if reference is not None:                            # comparison with the annotated reference
        log('  ' + reference.compare(name, image, roi, st['final_mask'], run_dir))
    log('')
    return polys


def main(show=True):
    from matplotlib import pyplot as plt
    run_dir = new_run_dir(LAB)
    lines = []

    def log(text):
        print(text)
        lines.append(text)

    reference = None
    try:
        from evaluate_reference import Reference
        reference = Reference()
    except Exception as e:                               # reference is optional
        log(f'(reference comparison skipped: {e})')

    for name in IMAGES:
        process(name, run_dir, log, reference)

    with open(os.path.join(run_dir, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Results saved to {os.path.abspath(run_dir)}')

    if show:
        plt.show()                                       # all windows at once
    plt.close('all')


if __name__ == '__main__':
    main(show='--no-show' not in sys.argv)
