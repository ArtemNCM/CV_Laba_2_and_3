# ----------------  Lab 3: R&D - вибір комбінації фільтрації та корекції кольору  ---------------

'''
Research script (slow, ~5-10 min): which filtering / colour / histogram operations
help building detection, and which settings make the building count match the real one.

 1. Image quality of every enhancement operation (brightness, contrast, entropy, sharpness)
 2. Every enhancement chain + both detectors (lab2 - simple, lab3 - full) on both photos,
    scored against the reference annotation (precision, recall, IoU, buildings found)
 3. Count calibration: with the chosen chain, the minimal building area is scanned so that
    the number of buildings found on the photo matches the real number (31, reference)

Results -> output/runX_lab3_research: research.txt, research.csv, research_iou.png

Run:
    python research.py
'''

import os
import sys
import csv
import importlib.util
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..')]
from common import IMAGES, load, new_run_dir          # noqa: E402
from evaluate_reference import Reference, evaluate     # noqa: E402
import enhancement                                     # noqa: E402

CHAINS = [[], ['blur'], ['gaussian'], ['median'], ['bilateral'], ['sharpen'],
          ['gray_world'], ['dehaze'], ['gamma'], ['equalize'], ['clahe'],
          ['median', 'sharpen'], ['sharpen', 'median'], ['dehaze', 'median'], ['dehaze', 'sharpen'],
          ['equalize', 'median'], ['clahe', 'median'], ['clahe', 'bilateral'],
          ['dehaze', 'clahe', 'median'], ['median', 'clahe', 'sharpen']]
CHOSEN = ['clahe', 'median']
LANDSAT, BING = 'KPI_esri', 'KPI'      # Esri Living Atlas (Landsat Explorer) and Bing photos
REAL_COUNT = 31                        # buildings in KPI_esri_annotated_reference.png


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scores(text):
    # parse the line returned by evaluate_reference.evaluate
    get = lambda key: float(text.split(key + '=')[1].split()[0])
    found = text.split('found ')[1].split()[0]
    return dict(precision=get('precision'), recall=get('recall'), iou=get('IoU'), found=found)


def main():
    run_dir = new_run_dir('lab3_research')
    lines = []

    def log(text=''):
        print(text, flush=True)
        lines.append(text)

    detectors = {'lab2': load_module('det_lab2', os.path.join(HERE, '..', 'lab2', 'buildings_detection.py')),
                 'lab3': load_module('det_lab3', os.path.join(HERE, 'buildings_detection.py'))}
    ref = Reference()
    data = {n: load(n) for n in IMAGES}
    gts = {n: cv2.warpPerspective(ref.mask, ref.H[n], im.shape[1::-1], flags=cv2.INTER_NEAREST)
           for n, (im, roi) in data.items()}

    def run(det, name, chain, **prm):
        image, roi = data[name]
        enhanced, _ = enhancement.enhance(image, chain, roi)
        params = dict(det.PARAMS, **prm)
        params.pop('enhance', None)
        if 'enhance' in det.PARAMS:
            params['enhance'] = []                       # enhancement already applied here
        polys, st = det.detect_buildings(enhanced, roi, params)
        return len(polys), scores(evaluate(name, st['final_mask'], gts[name], roi))

    # ---------------- 1. image quality ----------------
    log('1. IMAGE QUALITY after each operation (inside the ROI)')
    for name, (image, roi) in data.items():
        log(f'  {name}')
        for op in ['original'] + list(enhancement.OPERATIONS):
            img = image if op == 'original' else enhancement.OPERATIONS[op](image, roi)
            q = enhancement.quality(img, roi)
            log(f'    {op:11s} ' + '  '.join(f'{k} {v:7.1f}' for k, v in q.items()))
    log()

    # ---------------- 2. chains x detectors ----------------
    log('2. DETECTION with each enhancement chain (score vs reference annotation)')
    rows = []
    for chain in CHAINS:
        for det_name, det in detectors.items():
            for name in data:
                n, sc = run(det, name, chain)
                rows.append(dict(chain=' + '.join(chain) or 'none', detector=det_name, image=name,
                                 count=n, **sc))
        last = rows[-2 * len(data):]
        log(f"  {rows[-1]['chain']:28s} " + ' | '.join(
            f"{r['detector']} {r['image']:8s} n={r['count']:2d} IoU={r['iou']:.2f} found={r['found']}"
            for r in last))
    with open(os.path.join(run_dir, 'research.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log()
    for det_name in detectors:
        best = max(CHAINS, key=lambda ch: sum(r['iou'] for r in rows if r['detector'] == det_name
                                              and r['chain'] == (' + '.join(ch) or 'none')))
        log(f'  best chain for {det_name} detector: {" + ".join(best) or "none"}')
    log(f'  chain used in lab3/buildings_detection.py: {" + ".join(CHOSEN)}')
    log()

    # ---------------- 3. count calibration ----------------
    log(f'3. COUNT CALIBRATION (lab3 detector, {" + ".join(CHOSEN)}): real number of buildings = {REAL_COUNT}')
    det = detectors['lab3']
    results = {}

    def scan(values):
        for min_area in values:
            if min_area in results:
                continue
            results[min_area] = {name: run(det, name, CHOSEN, min_area=min_area) for name in data}
            log(f'  min_area={min_area:3d}  ' + '  '.join(
                f"{name}: n={n:2d} IoU={sc['iou']:.2f}" for name, (n, sc) in results[min_area].items()))

    scan(range(150, 401, 25))                            # coarse scan ...
    coarse = min(results, key=lambda a: abs(results[a][LANDSAT][0] - REAL_COUNT))
    scan(range(coarse - 20, coarse + 25, 5))             # ... then fine scan around the best value
    best_diff = min(abs(r[LANDSAT][0] - REAL_COUNT) for r in results.values())
    good = sorted(a for a, r in results.items() if abs(r[LANDSAT][0] - REAL_COUNT) == best_diff)
    min_area = good[len(good) // 2]                      # middle of the range that gives the count
    res = results[min_area]
    log(f'  min_area values giving the closest count: {good}')
    log(f'  -> min_area={min_area}: {LANDSAT} (Landsat Explorer) {res[LANDSAT][0]} buildings, '
        f'{BING} (Bing) {res[BING][0]} buildings, reference {REAL_COUNT}')
    log()

    # ---------------- chart ----------------
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, len(data), figsize=(16, 7), sharey=True)
    for ax, name in zip(axes, data):
        for k, det_name in enumerate(detectors):
            vals = [next(r['iou'] for r in rows if r['detector'] == det_name and r['image'] == name
                         and r['chain'] == (' + '.join(ch) or 'none')) for ch in CHAINS]
            ax.barh(np.arange(len(CHAINS)) + 0.4 * k, vals, height=0.4, label=f'{det_name} detector')
        ax.set_yticks(np.arange(len(CHAINS)) + 0.2)
        ax.set_yticklabels([' + '.join(ch) or 'none' for ch in CHAINS], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'{name}: IoU with the reference')
        ax.set_xlabel('IoU')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(run_dir, 'research_iou.png'), dpi=100)

    with open(os.path.join(run_dir, 'research.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Results saved to {os.path.abspath(run_dir)}')


if __name__ == '__main__':
    main()
