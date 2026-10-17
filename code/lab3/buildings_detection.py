# ----------------  Lab 3: виявлення будівель з попередньою обробкою зображення (KPI, KPI_esri)  ---------------

'''
Algorithm (classical CV, no neural networks):

 0. Image enhancement  - brightness histogram correction (CLAHE on Lab-L) + median filter
                         (chosen by research.py out of filtering / colour / histogram
                         operations in enhancement.py)
 1. Preprocessing      - mean-shift filtering (flattens roof texture, keeps edges)
 2. Colour features    - HSV + Lab; roofs are grey/bluish (low S, low Lab-b),
                         roads/paths are warmer (higher Lab-b), vegetation is saturated
 3. Shadow mask        - globally dark pixels + pixels much darker than their surroundings
 4. Seeds              - strict roof mask (very likely roof pixels)
 5. Region growing     - seeds are grown inside a looser roof mask (limited number of steps)
 6. Morphology         - opening / closing / hole filling
 7. Splitting          - watershed on the distance transform separates buildings that
                         are glued together by thin roads; non-compact pieces are split
                         again recursively with thicker markers
 8. Object filtering   - area, mean colour and the SHADOW TEST: the sun is in the south,
                         so every building casts a shadow to the north (up in the image);
                         flat objects (roads, squares) have no shadow
 9. Vectorization      - contours -> rotated rectangles or right-angle polygons
                         (buildings are rectangular), numbering, counting

Experimental rules for the hardest reference buildings (PARAMS fix_green, fix_dark,
fix_warm) are implemented but switched OFF: measured against the reference, none of them
was a real improvement (green roof #7/#12 is not separable from trees by colour or
smoothness; dark roof #24 looks exactly like a shadow; warm roofs #21/#25 are found but
roofs then grow into squares and paths). See the comments next to the parameters.

Package            Version
------------------ -----------
opencv-python      >= 4.8
numpy              >= 1.25
matplotlib         >= 3.7

Run:
    python buildings_detection.py             # process both photos, show results,
                                              # save everything to output/runX_lab3
    python buildings_detection.py --no-show   # only save, no windows

Every run also detects buildings WITHOUT enhancement, so report.txt shows the effect of
the enhancement (before / after) and compares the counts on both photos.
'''

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import (IMAGES, load, fill_holes, new_run_dir,  # noqa: E402
                    ellipse, draw_buildings)
import enhancement                                           # noqa: E402

LAB = 'lab3'
REFERENCE_COUNT = 31             # buildings marked in KPI_esri_annotated_reference.png

# Parameters are tuned for KPI_esri (800 px wide); pixel sizes are scaled by image width.
PARAMS = dict(
    enhance=['clahe', 'median'],   # image enhancement chain (see enhancement.py, research.py)
    base_width=800,
    roi_erode=11,          # px, cut the road that runs along the ROI border
    shadow_pct=15,         # % darkest pixels of ROI are shadow
    shadow_local=20,       # or: darker than local mean by this value ...
    shadow_local_pct=40,   # ... and darker than this percentile
    shadow_sigma=25,       # px, local mean window
    roof_dl=15,            # roof must be brighter than shadow threshold by this
    seed_s_max=70,         # strict mask: saturation
    seed_b_pct=40,         # strict mask: Lab-b below this percentile (grey-blue roofs)
    grow_s_max=90,         # loose mask: saturation
    grow_b_pct=70,         # loose mask: Lab-b percentile
    grow_l_pct=50,         # loose mask: brightness percentile
    grow_steps=6,          # region growing iterations (3x3)
    close_size=5,
    core_dist=3.0,         # px, distance-transform level for watershed markers (narrow buildings)
    core_big=6.0,          # px, same for wide buildings
    min_solidity=0.5,      # less compact pieces are split again
    min_area=240,          # px^2, calibrated by research.py so the count matches the real one
    obj_b_pct=50,          # object mean Lab-b must be below this percentile
    north_dist=10,         # px, height of the strip above the object checked for shadow
    shadow_ratio=0.3,      # min share of shadow pixels in that strip
    merge_ratio=1.0,       # glue watershed pieces if border >= ratio * piece width
    # --- rules for the hard cases of the reference (#7/#12 green, #24 dark, #21/#25 warm) ---
    # All OFF: tested against the reference, none of them is a real improvement (see report).
    # green: neither smoothness nor colour separates the patina roof from trees reliably
    #        (on KPI_esri the roof is even rougher than vegetation) -> IoU 0.52 -> 0.51
    fix_green=0,           # 1 = green-roof rule: greenish AND (smooth OR bluish), not plain hue
    green_h_min=56,        # hue range of the copper-patina roofs (vegetation ~50)
    green_b_pct=40,        # ... bluish: Lab-b below this percentile (vegetation is yellower)
    green_tex_pct=35,      # ... or smooth: local brightness std below this percentile
    texture_sigma=5,       # px, window of the local std (smoothness)
    # dark: #24 has the same brightness, colour and position (roof south of it) as the shadows
    #       of tall buildings -> it finds #24 but also adds the big shadow north of #11
    fix_dark=0,            # 1 = dark roofs: dark, bluish-black, compact, not just north of a roof
    dark_b_pct=8,          # dark roof: Lab-b below this percentile of the ROI
    dark_min_area=500,     # px^2, dark roofs are big compact blocks
    dark_south_roof=0.9,   # max share of roof pixels just south of it (else it is a shadow)
    # warm: finds #21 and #25 (24/24 reference blocks on KPI_esri), but roofs grow into
    #       squares, paths and trees -> big merged blobs; use with obj_b_pct=60
    fix_warm=0,            # 1 = warm (beige/brown) low-saturation pixels are allowed as roof seeds
    warm_s_max=45,
    warm_b_pct=65,
    rect_fill=0.75,        # object fills >= 75% of its rotated rectangle -> draw the rectangle
    grid=6,                # px, step of the right-angle outline (smaller = more detail)
    step_fill=1,           # 1 = remove small steps from the outline
)


def north_strip(mask, d):
    # pixels up to d px above the object (shadow side)
    k = np.zeros((2 * d + 1, 1), np.uint8)
    k[d:, 0] = 1
    return cv2.dilate(mask, k) & ~mask


def south_strip(mask, d):
    # pixels up to d px below the object (sun side)
    k = np.zeros((2 * d + 1, 1), np.uint8)
    k[:d + 1, 0] = 1
    return cv2.dilate(mask, k) & ~mask


def scaled(prm, width):
    # convert pixel parameters to the actual image resolution
    s = width / prm['base_width']
    p = dict(prm)
    for k in ['roi_erode', 'shadow_sigma', 'grow_steps', 'north_dist', 'texture_sigma']:
        p[k] = max(1, int(round(prm[k] * s)))
    p['core_dist'] = prm['core_dist'] * s
    p['core_big'] = prm['core_big'] * s
    p['min_area'] = prm['min_area'] * s * s
    p['dark_min_area'] = prm['dark_min_area'] * s * s
    p['grid'] = max(2, int(round(prm['grid'] * s)))
    return p


# --------------------------- 2-3. colour features + shadows ------------------
def image_features(image, roi, p):
    smooth = cv2.pyrMeanShiftFiltering(image, 7, 20)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.int32)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.int32)
    f = dict(smooth=smooth, H=hsv[..., 0], S=hsv[..., 1], L=lab[..., 0], B=lab[..., 2])

    L = f['L']
    t_dark = np.percentile(L[roi], p['shadow_pct'])
    L_local = cv2.GaussianBlur(L.astype(np.float32), (0, 0), p['shadow_sigma'])
    f['shadow'] = (L < t_dark) | ((L_local - L > p['shadow_local']) &
                                  (L < np.percentile(L[roi], p['shadow_local_pct'])))
    f['t_dark'] = t_dark

    # smoothness: local standard deviation of brightness (low = smooth surface)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), p['texture_sigma'])
    f['texture'] = np.sqrt(np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), p['texture_sigma']) - mean * mean, 0))

    return f


# --------------------------- 4-6. roof segmentation ---------------------------
def roof_mask(f, roi, p):
    H, S, L, B = f['H'], f['S'], f['L'], f['B']
    roi_in = cv2.erode(roi.astype(np.uint8) * 255, ellipse(p['roi_erode'])) > 0
    bright = L > f['t_dark'] + p['roof_dl']

    # green (copper-patina) roofs of the main building
    if p['fix_green']:
        # greenish, AND either bluish (vegetation is yellower) or smooth (vegetation is rough);
        # which of the two works depends on the photo (see analysis)
        greenish = (H >= p['green_h_min']) & (H <= 95) & (S > 40) & (S < 160) & bright
        bluish = B < np.percentile(B[roi], p['green_b_pct'])
        smooth = f['texture'] < np.percentile(f['texture'][roi], p['green_tex_pct'])
        green_roof = greenish & (bluish | smooth)
    else:
        green_roof = (H >= 60) & (H <= 95) & (S > 50) & (S < 160) & bright

    grey_seed = (S < p['seed_s_max']) & (B < np.percentile(B[roi], p['seed_b_pct']))
    if p['fix_warm']:                                    # beige / brown roofs: low saturation, warm
        grey_seed |= (S < p['warm_s_max']) & (B < np.percentile(B[roi], p['warm_b_pct']))
    strict = (grey_seed & bright) | green_roof
    loose = ((S < p['grow_s_max']) & (B < np.percentile(B[roi], p['grow_b_pct'])) &
             (L > np.percentile(L[roi], p['grow_l_pct'])) & ~f['shadow']) | green_roof

    strict = (strict & roi_in).astype(np.uint8) * 255
    loose = (loose & roi_in).astype(np.uint8) * 255

    seeds = cv2.morphologyEx(strict, cv2.MORPH_OPEN, ellipse(3))
    grown = seeds.copy()
    for _ in range(p['grow_steps']):                     # geodesic dilation inside loose mask
        grown = (cv2.dilate(grown, ellipse(3)) & loose) | seeds

    m = cv2.morphologyEx(grown, cv2.MORPH_OPEN, ellipse(3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse(p['close_size']))
    m = fill_holes(m)
    return seeds, m


# --------------------------- 6b. dark roofs ---------------------------------
def dark_roofs(f, roi, roofs, p):
    # Dark roofs look like shadows. Difference: a shadow lies directly NORTH of the building
    # that casts it, so a bright roof touches the shadow from the south. A dark roof has no
    # roof on its south side, and it is bluish-black while shadows on grass/trees are greenish.
    if not p['fix_dark']:
        return []
    roi_in = cv2.erode(roi.astype(np.uint8) * 255, ellipse(p['roi_erode'])) > 0
    dark = f['shadow'] & (f['B'] < np.percentile(f['B'][roi], p['dark_b_pct'])) & roi_in
    dark = cv2.morphologyEx(dark.astype(np.uint8) * 255, cv2.MORPH_OPEN, ellipse(5))
    dark = fill_holes(cv2.morphologyEx(dark, cv2.MORPH_CLOSE, ellipse(5)))
    found = []
    for piece in watershed_pieces(dark, p['core_dist'], p['core_big']):
        if cv2.countNonZero(piece) < p['dark_min_area'] or solidity(piece) < 0.6:
            continue
        south = south_strip(piece, p['north_dist']) > 0
        if (roofs[south] > 0).mean() < p['dark_south_roof']:      # no roof casting it -> a roof
            found.append(piece)
    return found


# --------------------------- 7. watershed splitting --------------------------
def watershed_pieces(mask, core_small, core_big):
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    # two-level markers: thick cores split big blobs, thin cores keep narrow buildings
    thick = (dt > core_big).astype(np.uint8)
    thin = (dt > core_small).astype(np.uint8)
    nt, lt = cv2.connectedComponents(thin)
    has_thick = np.unique(lt[thick > 0])
    cores = thick | np.isin(lt, has_thick, invert=True).astype(np.uint8) & thin
    n, markers = cv2.connectedComponents(cores)
    markers = markers.astype(np.int32) + 1               # background = 1
    markers[(mask > 0) & (cores == 0)] = 0               # unknown zone
    relief = cv2.cvtColor(np.uint8(255 - np.clip(dt * 10, 0, 255)), cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(relief, markers)
    return [((markers == i) & (mask > 0)).astype(np.uint8) * 255 for i in range(2, n + 1)]


def solidity(piece):
    c = max(cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    return cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)


def split_objects(mask, p, depth=0):
    # big non-compact pieces (buildings glued by squares/roads) are split again
    # with thicker markers
    k = 1.5 ** depth
    pieces = []
    for piece in watershed_pieces(mask, p['core_dist'] * k, p['core_big'] * k):
        if cv2.countNonZero(piece) == 0:
            continue
        if depth < 3 and cv2.countNonZero(piece) > 6 * p['min_area'] and solidity(piece) < p['min_solidity']:
            sub = split_objects(piece, p, depth + 1)
            if len(sub) > 1:
                pieces += sub
                continue
        pieces.append(piece)
    return pieces


# --------------------------- 8. object filtering -----------------------------
def filter_objects(candidates, f, roi, p, extra=()):
    t_b = np.percentile(f['B'][roi], p['obj_b_pct'])
    shadow = f['shadow'].astype(np.uint8)
    pieces, rejected = [], []                            # rejected: (piece, reason)
    for comp in candidates:
        comp = cv2.morphologyEx(comp, cv2.MORPH_OPEN, ellipse(3))
        if cv2.countNonZero(comp) == 0:
            continue
        if cv2.countNonZero(comp) < p['min_area']:
            rejected.append((comp, 'small'))
            continue
        if f['B'][comp > 0].mean() > t_b:                # too warm -> road / sand / court
            rejected.append((comp, 'colour'))
            continue
        strip = north_strip(comp, p['north_dist']) > 0
        if shadow[strip].mean() < p['shadow_ratio']:     # no shadow on the north side -> flat object
            rejected.append((comp, 'no_shadow'))
            continue
        pieces.append(comp)
    pieces += list(extra)                                # already verified objects (dark roofs)
    return merge_pieces(pieces, p), rejected


def merge_pieces(pieces, p):
    # watershed may cut one building into wings; glue neighbours whose common border is
    # nearly as wide as the pieces themselves (a thin neck = road between two buildings)
    k = len(pieces)
    parent = list(range(k))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    width = [2 * cv2.distanceTransform(c, cv2.DIST_L2, 5).max() for c in pieces]
    grown = [cv2.dilate(c, ellipse(7)) for c in pieces]
    for i in range(k):
        for j in range(i + 1, k):
            contact = cv2.countNonZero(grown[i] & pieces[j])
            if contact and contact >= p['merge_ratio'] * min(width[i], width[j]):
                parent[root(j)] = root(i)

    objects = np.zeros(pieces[0].shape if pieces else (1, 1), np.int32)
    ids = {}
    for i, c in enumerate(pieces):
        objects[c > 0] = ids.setdefault(root(i), len(ids) + 1)
    return objects


# --------------------------- 9. vectorization -------------------------------
def wall_direction(contour):
    # main wall direction (degrees, modulo 90): histogram of the outline's edge
    # directions weighted by edge length
    poly = cv2.approxPolyDP(contour, 2, True).reshape(-1, 2).astype(np.float64)
    d = np.roll(poly, -1, axis=0) - poly
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90
    hist = np.bincount(ang.astype(int) % 90, weights=np.hypot(d[:, 0], d[:, 1]), minlength=90)
    hist = np.convolve(np.concatenate([hist[-3:], hist, hist[:3]]), np.ones(7), 'valid')  # circular smoothing
    return float(hist.argmax())


def regularize(contour, p):
    # buildings consist of rectangles: replace the ragged outline by
    #  - its minimal rotated rectangle, if the object fills it well, or
    #  - a polygon with right angles only, aligned with the building's main direction
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    if cv2.contourArea(contour) >= p['rect_fill'] * w * h:
        return np.int32(cv2.boxPoints(rect)).reshape(-1, 1, 2)

    # draw the object on a square canvas around its centre and rotate it so that
    # its walls become horizontal / vertical
    g = p['grid']
    size = (int(np.hypot(w, h)) // g + 4) * g             # multiple of the grid step
    shift = np.array([size / 2 - cx, size / 2 - cy])
    canvas = np.zeros((size, size), np.uint8)
    cv2.drawContours(canvas, [np.int32(contour.reshape(-1, 2) + shift)], -1, 255, -1)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), wall_direction(contour), 1.0)
    rot = cv2.warpAffine(canvas, M, (size, size), flags=cv2.INTER_NEAREST)

    # snap to a coarse grid -> only horizontal/vertical edges
    small = cv2.resize(rot, (size // g, size // g), interpolation=cv2.INTER_AREA)
    small = np.where(small >= 128, 255, 0).astype(np.uint8)
    if p['step_fill']:                                   # fill one-cell notches (small steps)
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    blocky = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
    cnts = cv2.findContours(blocky, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cnts:
        return np.int32(cv2.boxPoints(rect)).reshape(-1, 1, 2)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)

    # rotate the corners back to image coordinates
    Mi = cv2.invertAffineTransform(M)
    back = c @ Mi[:, :2].T + Mi[:, 2] - shift
    return np.int32(np.round(back)).reshape(-1, 1, 2)


def image_contours(objects, p):
    polys = []
    for i in range(1, objects.max() + 1):
        m = (objects == i).astype(np.uint8) * 255
        m = fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse(5)))
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(c) < p['min_area']:
                continue
            polys.append(regularize(c, p))
    # number buildings top-to-bottom, left-to-right (rows ~ 40 px)
    polys.sort(key=lambda c: (int(cv2.boundingRect(c)[1] // 40), cv2.boundingRect(c)[0]))
    return polys


REJECT_COLOURS = {                                     # BGR
    'small': (0, 255, 255),        # yellow  - smaller than min_area
    'colour': (255, 160, 0),       # blue    - too warm/yellow (road, sand, court)
    'no_shadow': (255, 0, 255),    # magenta - no shadow on the north side
}


def draw_debug(image, objects, rejected, shadow):
    # why was each piece of the roof mask kept or thrown away
    out = (image * 0.6).astype(np.uint8)
    out[shadow > 0] = (out[shadow > 0] * 0.4).astype(np.uint8)          # shadows darker
    layer = out.copy()
    for comp, reason in rejected:
        layer[comp > 0] = REJECT_COLOURS[reason]
    layer[objects > 0] = (0, 0, 255)                                     # accepted - red
    out = cv2.addWeighted(out, 0.45, layer, 0.55, 0)
    for comp, reason in rejected:
        cnts = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(out, cnts, -1, REJECT_COLOURS[reason], 1)
    legend = [('accepted', (0, 0, 255)), ('rejected: too small', REJECT_COLOURS['small']),
              ('rejected: colour (road/sand)', REJECT_COLOURS['colour']),
              ('rejected: no shadow to north', REJECT_COLOURS['no_shadow'])]
    cv2.rectangle(out, (5, 5), (255, 12 + 18 * len(legend)), (0, 0, 0), -1)
    for i, (text, col) in enumerate(legend):
        y = 20 + 18 * i
        cv2.rectangle(out, (10, y - 9), (22, y + 3), col, -1)
        cv2.putText(out, text, (28, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return out


# --------------------------- full pipeline ----------------------------------
def detect_buildings(image, roi, prm=PARAMS):
    p = scaled(prm, image.shape[1])
    enhanced, enh_steps = enhancement.enhance(image, prm['enhance'], roi)
    f = image_features(enhanced, roi, p)
    seeds, mask = roof_mask(f, roi, p)
    candidates = split_objects(mask, p)
    darks = dark_roofs(f, roi, mask, p)
    objects, rejected = filter_objects(candidates, f, roi, p, darks)
    polys = image_contours(objects, p)
    final = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(final, polys, -1, 255, -1)
    stages = dict(enhanced=enhanced, enh_steps=enh_steps,
                  smooth=f['smooth'], shadow=f['shadow'].astype(np.uint8) * 255,
                  seeds=seeds, roof_mask=mask, final_mask=final,
                  debug=draw_debug(image, objects, rejected, f['shadow']))
    return polys, stages


def stages_figure(name, st, result):
    from matplotlib import pyplot as plt
    titles = ['Enhanced: ' + ' + '.join(PARAMS['enhance']), 'Shadows', 'Roof seeds', 'Roof mask',
              'Kept / rejected pieces', 'Buildings']
    imgs = [st['enhanced'], st['shadow'], st['seeds'], st['roof_mask'], st['debug'], result]
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
    cv2.imwrite(os.path.join(run_dir, f'{name}_enhanced.png'), st['enhanced'])
    cv2.imwrite(os.path.join(run_dir, f'{name}_debug_rejected.png'), st['debug'])
    for k in ['smooth', 'shadow', 'seeds', 'roof_mask']:
        cv2.imwrite(os.path.join(run_dir, f'{name}_step_{k}.png'), st[k])
    fig = stages_figure(name, st, result)
    fig.savefig(os.path.join(run_dir, f'{name}_stages.png'), dpi=100)
    fig = enhancement.histogram_figure(f'{name}: image enhancement', st['enh_steps'], roi)
    fig.savefig(os.path.join(run_dir, f'{name}_histograms.png'), dpi=100)

    # the same detection without enhancement -> effect of Lab 3 preprocessing
    raw_polys, raw_st = detect_buildings(image, roi, dict(PARAMS, enhance=[]))
    cv2.imwrite(os.path.join(run_dir, f'{name}_buildings_no_enhancement.png'),
                draw_buildings(image, raw_polys, roi))

    q0, q1 = enhancement.quality(image, roi), enhancement.quality(st['enhanced'], roi)
    log(f'{name}: found {len(polys)} buildings (without enhancement: {len(raw_polys)})')
    log('  image quality      ' + '  '.join(f'{k}: {q0[k]:.1f} -> {q1[k]:.1f}' for k in q0))
    for i, c in enumerate(polys, 1):
        x, y, w, h = cv2.boundingRect(c)
        log(f'  #{i:2d}  center=({x + w // 2:3d},{y + h // 2:3d})  area={int(cv2.contourArea(c)):6d} px  vertices={len(c)}')

    if reference is not None:                            # comparison with the annotated reference
        log('  with enhancement:    ' + reference.compare(name, image, roi, st['final_mask'], run_dir))
        log('  without enhancement: ' + reference.compare(name, image, roi, raw_st['final_mask'], None))
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

    polys = {name: process(name, run_dir, log, reference) for name in IMAGES}
    log('Building count: ' + ', '.join(f'{n} = {len(p)}' for n, p in polys.items()) +
        f'  (reference annotation: {REFERENCE_COUNT})')

    if reference is not None:                            # same building on both photos
        import object_matching
        (ie, re), (ik, rk) = load('KPI_esri'), load('KPI')
        versions = {name: (enhancement.enhance(ie, chain, re)[0], enhancement.enhance(ik, chain, rk)[0])
                    for name, chain in [('raw', []), ('enhanced', PARAMS['enhance']),
                                        ('gray_world', ['gray_world'])]}
        H = reference.H['KPI'] @ np.linalg.inv(reference.H['KPI_esri'])
        log('')
        for line in object_matching.report_lines(
                *object_matching.compare(polys['KPI_esri'], polys['KPI'], ie, ik, H, versions)):
            log(line)

    with open(os.path.join(run_dir, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Results saved to {os.path.abspath(run_dir)}')

    if show:
        plt.show()                                       # all windows at once
    plt.close('all')


if __name__ == '__main__':
    main(show='--no-show' not in sys.argv)
