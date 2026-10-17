# ----------------  Lab 2: R&D - корекція кольору та векторизація  ---------------

'''
R&D for lab 2: which combination of colour correction and vectorization separates the
buildings best.

 1. Colour correction (formulas as in the lecture example filtr_im_PIL.py):
      grayscale  S = (R + G + B) / 3
      sepia      R' = S + 2*depth, G' = S + depth, B' = S   (depth = 20)
      negative   R' = 255 - R, G' = 255 - G, B' = 255 - B
      other      HSV saturation S (roofs are grey), Lab b* (roofs bluish, roads yellowish),
                 and the combined HSV + Lab mask used in buildings_detection.py
    For every correction: how well its channel separates roof pixels from the rest
    (AUC, 0.5 = no separation, 1.0 = perfect) inside the region of interest.

 2. Vectorization: the mask of every correction (Otsu threshold for single channels) is
    vectorized by
      Contours   - findContours of the mask only
      Canny      - the mask is cut by Canny edges, then contours
      Gabor      - the mask is cut by Gabor filter bank edges, then contours
    and the buildings are scored against the reference annotation.

Results -> output/runX_lab2_research: research.txt, research.csv, figures

Run:
    python research.py
'''

import os
import sys
import csv
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..')]
import buildings_detection as det                      # noqa: E402
from common import IMAGES, load, new_run_dir, ellipse  # noqa: E402
from evaluate_reference import Reference, evaluate     # noqa: E402

SEPIA_DEPTH = 20
VECTORIZATION = {'Contours': 'none', 'Canny': 'canny', 'Gabor': 'gabor', 'Canny + Gabor': 'both'}


# --------------------------- colour corrections (BGR images) ------------------
def grayscale(img):
    s = img.astype(np.int32).sum(axis=2) // 3
    return np.dstack([s, s, s]).astype(np.uint8)


def sepia(img, depth=SEPIA_DEPTH):
    s = img.astype(np.int32).sum(axis=2) // 3
    r, g, b = s + 2 * depth, s + depth, s
    return np.clip(np.dstack([b, g, r]), 0, 255).astype(np.uint8)


def negative(img):
    return 255 - img


CORRECTIONS = {
    # name: (corrected image for the figure, single channel used for the mask)
    'grayscale': lambda im: (grayscale(im), grayscale(im)[..., 0]),
    'sepia': lambda im: (sepia(im), cv2.cvtColor(sepia(im), cv2.COLOR_BGR2GRAY)),
    'negative': lambda im: (negative(im), grayscale(negative(im))[..., 0]),
    'HSV saturation': lambda im: (cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1], cv2.COLOR_GRAY2BGR),
                                  cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1]),
    'Lab b*': lambda im: (cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 2], cv2.COLOR_GRAY2BGR),
                          cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 2]),
}


def auc(channel, roof, rest):
    # probability that a random roof pixel is brighter than a random other pixel (via histograms)
    h1 = np.bincount(channel[roof].ravel(), minlength=256).astype(np.float64)
    h0 = np.bincount(channel[rest].ravel(), minlength=256).astype(np.float64)
    h1, h0 = h1 / h1.sum(), h0 / h0.sum()
    below = np.cumsum(h0) - h0
    return float((h1 * (below + 0.5 * h0)).sum())


def otsu_mask(channel, roi, roofs_brighter, p):
    t, _ = cv2.threshold(channel[roi].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = channel > t if roofs_brighter else channel <= t
    roi_in = cv2.erode(roi.astype(np.uint8) * 255, ellipse(p['roi_erode'])) > 0
    return (m & roi_in).astype(np.uint8) * 255, t


def vectorize_mask(image, roi, mask, edges_mode, p):
    prm = dict(p, edges=edges_mode)
    edges, _ = det.find_edges(image, roi, prm)
    clean = det.clean_mask(mask, edges, prm)
    _, feats = det.colour_mask(image, roi, prm)
    kept, _ = det.filter_contours(det.image_contours(clean), feats, roi, prm)
    polys = det.vectorize(kept, prm)
    final = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(final, polys, -1, 255, -1)
    return polys, final


def scores(text):
    get = lambda k: float(text.split(k + '=')[1].split()[0])
    return dict(precision=get('precision'), recall=get('recall'), iou=get('IoU'),
                found=text.split('found ')[1].split()[0])


def main():
    from matplotlib import pyplot as plt
    run_dir = new_run_dir('lab2_research')
    lines = []

    def log(text=''):
        print(text, flush=True)
        lines.append(text)

    ref = Reference()
    rows = []
    figs = {}
    for name in IMAGES:
        image, roi = load(name)
        p = det.scaled(det.PARAMS, image.shape[1])
        gt = cv2.warpPerspective(ref.mask, ref.H[name], image.shape[1::-1], flags=cv2.INTER_NEAREST) > 0
        roof = cv2.erode((gt & roi).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        rest = roi & ~gt
        log(f'{name}')
        log('  1. COLOUR CORRECTION: separation of roof pixels from the rest (AUC; 0.5 = none, 1 = perfect)')
        masks, panels = {}, [('original', image)]
        for cname, fn in CORRECTIONS.items():
            corrected, channel = fn(image)
            a = auc(channel, roof, rest)
            sep = max(a, 1 - a)
            masks[cname], t = otsu_mask(channel, roi, a >= 0.5, p)
            panels.append((cname, corrected))
            log(f'    {cname:15s} AUC {a:.3f}  separation {sep:.3f}  '
                f'(roofs {"brighter" if a >= 0.5 else "darker"}, Otsu threshold {t:.0f})')
            rows.append(dict(image=name, correction=cname, auc=round(a, 3), separation=round(sep, 3)))
        mask_sel, _ = det.colour_mask(image, roi, p)
        masks['HSV + Lab (selected)'] = mask_sel
        sel = (mask_sel > 0)
        log(f'    {"HSV + Lab mask":15s} share of roof pixels in the mask {sel[roof].mean():.3f}, '
            f'of other pixels {sel[rest].mean():.3f}')
        figs[name] = panels

        log('  2. VECTORIZATION of every mask: IoU with the reference (reference blocks found, buildings counted)')
        for cname, mask in masks.items():
            res = []
            for vname, mode in VECTORIZATION.items():
                polys, final = vectorize_mask(image, roi, mask, mode, p)
                sc = scores(evaluate(name, final, gt.astype(np.uint8) * 255, roi))
                res.append(f'{vname} {sc["iou"]:.2f} ({sc["found"]}, n={len(polys)})')
                rows.append(dict(image=name, correction=cname, vectorization=vname, count=len(polys), **sc))
            log(f'    {cname:21s} ' + ' | '.join(res))
        log()

    # ---------------- figures
    for name, panels in figs.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        for ax, (t, im) in zip(axes.ravel(), panels):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(t), ax.set_xticks([]), ax.set_yticks([])
        fig.suptitle(f'{name}: colour corrections')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(run_dir, f'{name}_corrections.png'), dpi=100)

    image, roi = load('KPI_esri')
    p = det.scaled(det.PARAMS, image.shape[1])
    canny = det.canny_edges(image, p)
    gab, resp = det.gabor_edges(image, roi, p)
    kernels = [cv2.getGaborKernel((p['gabor_ksize'], p['gabor_ksize']), p['gabor_sigma'], np.pi * k / 4,
                                  p['gabor_lambda'], p['gabor_gamma'], 0, ktype=cv2.CV_32F) for k in range(4)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    show = [('Canny edges (150 / 375)', canny), ('Gabor response (max of 8 orientations)', resp),
            ('Gabor edges (strongest 3 %)', gab)]
    for ax, (t, im) in zip(axes[0], show):
        ax.imshow(im, 'gray'), ax.set_title(t), ax.set_xticks([]), ax.set_yticks([])
    axes[1, 0].imshow(np.hstack([cv2.resize(k, (84, 84), interpolation=cv2.INTER_CUBIC) for k in kernels]), 'gray')
    axes[1, 0].set_title('Gabor kernels, θ = 0°, 45°, 90°, 135°'), axes[1, 0].set_xticks([]), axes[1, 0].set_yticks([])
    for ax, mode, t in [(axes[1, 1], 'canny', 'Mask cut by Canny'), (axes[1, 2], 'gabor', 'Mask cut by Gabor')]:
        mask, _ = det.colour_mask(image, roi, p)
        e, _ = det.find_edges(image, roi, dict(p, edges=mode))
        ax.imshow(det.clean_mask(mask, e, p), 'gray'), ax.set_title(t), ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle('KPI_esri: vectorization - Canny and Gabor')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(run_dir, 'KPI_esri_canny_gabor.png'), dpi=100)

    fields = ['image', 'correction', 'vectorization', 'count', 'precision', 'recall', 'iou', 'found', 'auc', 'separation']
    with open(os.path.join(run_dir, 'research.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(run_dir, 'research.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Results saved to {os.path.abspath(run_dir)}')


if __name__ == '__main__':
    main()
