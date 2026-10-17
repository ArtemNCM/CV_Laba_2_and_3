# ----------------  Виявлення будівель нейромережею UNet++ (KPI, KPI_esri)  ---------------

'''
Building detection with a pretrained neural network (deep learning), for comparison
with the classical algorithms in lab2 / lab3.

Model:  giswqs/whu-building-unetplusplus-efficientnet-b4  (Hugging Face, Apache-2.0)
        https://huggingface.co/giswqs/whu-building-unetplusplus-efficientnet-b4
        UNet++ segmentation network, EfficientNet-B4 encoder, 2 classes (background, building),
        trained on the WHU Building Dataset: aerial photos, 0.3 m per pixel, 512x512 tiles.

Pipeline:
 0. Enhancement - histogram equalization + sharpening (lab3/enhancement.py); the photos are
                  much darker and less contrasted than the training data - without this step
                  the model finds only a few buildings (see report.txt)
 1. Upscale     - our photos are ~2 m per pixel, the model was trained on 0.3 m per pixel;
                  2x enlargement worked best (1.5x, 3x, 4x were worse)
 2. Normalize   - BGR -> RGB, values / 255 (same as in the model's own inference code)
 3. Tiling      - 512x512 windows with 256 px overlap; predictions of overlapping windows
                  are blended with weights that fall towards the window edges
 4. Softmax     - probability of "building" for every pixel; back to the original size
 5. Threshold   - probability >= 0.3 inside the region of interest -> building mask
 6. Vectorize   - contours -> rotated rectangles / right-angle polygons (same as lab3),
                  numbering, counting, comparison with the reference annotation

The weights (model.pth, 84 MB) are downloaded once into the Hugging Face cache
(~/.cache/huggingface) and reused afterwards.

Package                      Version
---------------------------- -----------
torch                        >= 2.0
segmentation-models-pytorch  >= 0.3
huggingface_hub              >= 0.20
opencv-python, numpy, matplotlib

Run:
    python model_detection.py              # both photos, show results, save to output/runX_model
    python model_detection.py --no-show    # only save
    python model_detection.py --scale 3    # other upscaling factor
    python model_detection.py --finetuned  # weights from the newest output/runX_finetune
                                           # (made by finetune.py)
    python model_detection.py --weights path/to/model_finetuned.pth
'''

import os
import sys
import json
import time
import importlib.util
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, '..')
sys.path[:0] = [CODE, os.path.join(CODE, 'lab3')]
from common import IMAGES, OUTPUT, load, new_run_dir, ellipse, draw_buildings  # noqa: E402
import enhancement                                      # noqa: E402

LAB = 'model'
REPO_ID = 'giswqs/whu-building-unetplusplus-efficientnet-b4'
REFERENCE_COUNT = 31                                    # buildings in the reference annotation

ENHANCE = ['equalize', 'sharpen']   # image enhancement before the network (lab3/enhancement.py)
SCALE = 2              # upscaling factor (photo ~2 m/px, model trained on 0.3 m/px; 2x was best)
WINDOW = 512           # tile size used in training
OVERLAP = 256          # overlap between neighbouring tiles
THRESHOLD = 0.3        # building probability threshold
MIN_AREA = 120         # px^2 in the original photo (800 px wide), smaller objects are dropped


def load_lab3():
    # reuse vectorization / drawing of the classical algorithm (lab3)
    spec = importlib.util.spec_from_file_location('lab3_detection', os.path.join(CODE, 'lab3', 'buildings_detection.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab3 = load_lab3()


# --------------------------- model -------------------------------------------
def load_model(device, weights=None):
    # weights=None -> original pretrained weights from Hugging Face; else a local .pth file
    import torch
    import segmentation_models_pytorch as smp
    from huggingface_hub import hf_hub_download

    config = json.load(open(hf_hub_download(REPO_ID, 'config.json')))
    model = smp.UnetPlusPlus(encoder_name=config['encoder_name'], encoder_weights=None,
                             in_channels=config['num_channels'], classes=config['num_classes'])
    weights = weights or hf_hub_download(REPO_ID, 'model.pth')
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def window_weight(h, w):
    # 1 in the middle of the tile, falling to 0.1 at the edges (as in the model's inference code)
    y, x = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([x, w - x - 1, y, h - y - 1]).astype(np.float32)
    return np.maximum(np.minimum(edge, OVERLAP / 2) / (OVERLAP / 2), 0.1)


def predict_probability(model, image, device, scale=SCALE):
    import torch
    big = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = rgb.shape[:2]
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    stride = WINDOW - OVERLAP
    rows = range(0, max(H - OVERLAP, 1), stride)
    cols = range(0, max(W - OVERLAP, 1), stride)
    total, done, t0 = len(rows) * len(cols), 0, time.time()
    with torch.no_grad():
        for r in rows:
            for c in cols:
                tile = rgb[r:r + WINDOW, c:c + WINDOW]
                h, w = tile.shape[:2]
                padded = np.zeros((WINDOW, WINDOW, 3), np.float32)
                padded[:h, :w] = tile
                x = torch.from_numpy(padded.transpose(2, 0, 1)).unsqueeze(0).to(device)
                prob = torch.softmax(model(x), dim=1)[0, 1].cpu().numpy()[:h, :w]
                wt = window_weight(h, w)
                acc[r:r + h, c:c + w] += prob * wt
                wsum[r:r + h, c:c + w] += wt
                done += 1
                print(f'\r    tiles {done}/{total}  ({time.time() - t0:.0f} s)', end='', flush=True)
    print()
    prob = acc / np.maximum(wsum, 1e-6)
    return cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)


# --------------------------- vectorization -----------------------------------
def buildings_from_probability(prob, roi, image_width):
    p = lab3.scaled(dict(lab3.PARAMS, min_area=MIN_AREA), image_width)
    mask = ((prob >= THRESHOLD) & roi).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse(3))
    n, labels = cv2.connectedComponents(mask)
    polys = lab3.image_contours(labels, p)                 # rectangles / right-angle polygons
    final = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(final, polys, -1, 255, -1)
    return polys, mask, final


def heatmap(image, prob):
    colour = cv2.applyColorMap(np.uint8(np.clip(prob, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.5, colour, 0.5, 0)


def figure(name, image, prob, mask, result, scale, count):
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(15, 9))
    panels = [('Original', image), (f'Building probability (UNet++, x{scale})', heatmap(image, prob)),
              (f'Mask (p >= {THRESHOLD})', mask), ('Buildings', result)]
    for i, (t, im) in enumerate(panels, 1):
        plt.subplot(2, 2, i)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, 'gray')
        plt.title(t), plt.xticks([]), plt.yticks([])
    plt.suptitle(f'{name}: {count} buildings (neural network)')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def newest_finetuned():
    import re
    runs = [(int(m.group(1)), d) for d in os.listdir(OUTPUT) if (m := re.match(r'^run(\d+)_finetune$', d))
            and os.path.exists(os.path.join(OUTPUT, d, 'model_finetuned.pth'))]
    if not runs:
        raise FileNotFoundError('no output/runX_finetune/model_finetuned.pth - run finetune.py first')
    return os.path.join(OUTPUT, max(runs)[1], 'model_finetuned.pth')


# --------------------------- main --------------------------------------------
def main(show=True, scale=SCALE, weights=None):
    import torch
    from matplotlib import pyplot as plt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = new_run_dir(LAB)
    lines = []

    def log(text=''):
        print(text, flush=True)
        lines.append(text)

    log(f'Model {REPO_ID} on {device}, enhancement {" + ".join(ENHANCE)}, upscaling x{scale}, '
        f'tiles {WINDOW} px, overlap {OVERLAP} px, threshold {THRESHOLD}')
    log(f'Weights: {weights or "pretrained (Hugging Face)"}')
    model = load_model(device, weights)

    reference = None
    try:
        from evaluate_reference import Reference
        reference = Reference()
    except Exception as e:                               # reference is optional
        log(f'(reference comparison skipped: {e})')

    counts = {}
    for name in IMAGES:
        image, roi = load(name)
        log(f'{name}:')
        log(f'  without enhancement:')
        raw_prob = predict_probability(model, image, device, scale)
        _, _, raw_final = buildings_from_probability(raw_prob, roi, image.shape[1])
        log(f'  with enhancement ({" + ".join(ENHANCE)}):')
        enhanced, _ = enhancement.enhance(image, ENHANCE, roi)
        prob = predict_probability(model, enhanced, device, scale)
        polys, mask, final = buildings_from_probability(prob, roi, image.shape[1])
        result = draw_buildings(image, polys, roi)
        counts[name] = len(polys)

        cv2.imwrite(os.path.join(run_dir, f'{name}_buildings.png'), result)
        cv2.imwrite(os.path.join(run_dir, f'{name}_probability.png'), heatmap(image, prob))
        cv2.imwrite(os.path.join(run_dir, f'{name}_mask.png'), final)
        figure(name, image, prob, mask, result, scale, len(polys)).savefig(
            os.path.join(run_dir, f'{name}_stages.png'), dpi=100)

        log(f'  found {len(polys)} buildings')
        for i, c in enumerate(polys, 1):
            x, y, w, h = cv2.boundingRect(c)
            log(f'  #{i:2d}  center=({x + w // 2:3d},{y + h // 2:3d})  area={int(cv2.contourArea(c)):6d} px  vertices={len(c)}')
        if reference is not None:
            log('  with enhancement:    ' + reference.compare(name, image, roi, final, run_dir))
            log('  without enhancement: ' + reference.compare(name, image, roi, raw_final, None))
        log()

    log('Building count: ' + ', '.join(f'{n} = {c}' for n, c in counts.items()) +
        f'  (reference annotation: {REFERENCE_COUNT})')
    with open(os.path.join(run_dir, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Results saved to {os.path.abspath(run_dir)}')

    if show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    args = sys.argv[1:]
    s = float(args[args.index('--scale') + 1]) if '--scale' in args else SCALE
    w = args[args.index('--weights') + 1] if '--weights' in args else None
    if '--finetuned' in args:
        w = newest_finetuned()
    main(show='--no-show' not in args, scale=s, weights=w)
