# ----------------  Донавчання (fine-tuning) UNet++ на нашому знімку  ---------------

'''
Fine-tuning of the pretrained UNet++ (model_detection.py) on our own photo.

Fair test:
    TRAIN  KPI_esri (Landsat Explorer) + the reference annotation (KPI_esri_annotated_reference.png)
    TEST   KPI (Bing) - the network never sees this photo during training; its labels are only
           used at the end to score the result.

 1. Training data  - KPI_esri, enhanced like in model_detection.py (equalize + sharpen) and
                     upscaled 2x; labels = the reference outlines moved onto the photo with the
                     homography; pixels outside the region of interest are ignored by the loss
 2. Augmentation   - random 256x256 crops (half of them on buildings), flips, 90-degree turns,
                     scale, brightness / contrast / saturation / hue changes, blur and haze
                     (KPI is hazier and blurrier than KPI_esri - the network must not depend
                     on the exact colours of one photo)
 3. Training       - the EfficientNet-B4 encoder stays frozen (pretrained features, only one
                     training photo -> less overfitting, and 2x faster on CPU); the UNet++
                     decoder is trained with cross-entropy + Dice loss, AdamW, fixed number of
                     steps (no early stopping on the test photo)
 4. Evaluation     - pretrained vs fine-tuned network on both photos against the reference

Output -> output/runX_finetune: report.txt, loss curve, result images,
          model_finetuned.pth (weights for model_detection.py --weights ...)

Run (about 20-25 min on CPU):
    python finetune.py
    python finetune.py --steps 150     # shorter
'''

import os
import sys
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_detection as md                           # noqa: E402
from common import load, new_run_dir, draw_buildings   # noqa: E402
import enhancement                                     # noqa: E402

TRAIN, TEST = 'KPI_esri', 'KPI'
STEPS = 300
BATCH = 8
CROP = 256
LR = 3e-4
SEED = 0


# --------------------------- data --------------------------------------------
def training_data(reference):
    image, roi = load(TRAIN)
    gt = cv2.warpPerspective(reference.mask, reference.H[TRAIN], image.shape[1::-1],
                             flags=cv2.INTER_NEAREST) > 0
    enhanced, _ = enhancement.enhance(image, md.ENHANCE, roi)
    s = md.SCALE
    big = cv2.resize(enhanced, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    label = cv2.resize(gt.astype(np.uint8), big.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(roi.astype(np.uint8), big.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    label[valid == 0] = 255                              # ignored by the loss
    rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
    return rgb, label


def augment(img, lab, rng):
    # geometry
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    if rng.random() < 0.5:
        img, lab = img[::-1], lab[::-1]
    k = rng.integers(4)
    img, lab = np.rot90(img, k), np.rot90(lab, k)
    img = np.ascontiguousarray(img)
    lab = np.ascontiguousarray(lab)
    # colour: brightness, contrast, saturation, hue
    f = img.astype(np.float32)
    f = (f - f.mean()) * rng.uniform(0.7, 1.3) + f.mean() + rng.uniform(-30, 30)
    hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
    f = cv2.cvtColor(np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    # haze and blur (the test photo is hazier and less sharp)
    if rng.random() < 0.4:
        a = rng.uniform(0.1, 0.4)
        f = f * (1 - a) + a * rng.uniform(150, 210)
    if rng.random() < 0.4:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.5))
    return np.clip(f, 0, 255) / 255.0, lab


def batch(rgb, label, rng):
    H, W = label.shape
    xs, ys = [], []
    building = np.argwhere(label == 1)
    while len(xs) < BATCH:
        scale = rng.uniform(0.8, 1.25)
        size = int(CROP * scale)
        if rng.random() < 0.5:                           # half of the crops centred on a building
            cy, cx = building[rng.integers(len(building))]
            y0 = int(np.clip(cy - size // 2, 0, H - size))
            x0 = int(np.clip(cx - size // 2, 0, W - size))
        else:
            y0, x0 = rng.integers(0, H - size), rng.integers(0, W - size)
        lab = label[y0:y0 + size, x0:x0 + size]
        if (lab != 255).mean() < 0.3:                    # mostly outside the region of interest
            continue
        img = cv2.resize(rgb[y0:y0 + size, x0:x0 + size], (CROP, CROP), interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
        img, lab = augment(img, lab, rng)
        xs.append(img.transpose(2, 0, 1))
        ys.append(lab)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.int64)


# --------------------------- training ----------------------------------------
def dice_loss(logits, target):
    import torch
    prob = torch.softmax(logits, dim=1)[:, 1]
    valid = (target != 255).float()
    t = (target == 1).float() * valid
    p = prob * valid
    return 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


def train(model, rgb, label, steps, log, rng):
    import torch
    for prm in model.encoder.parameters():               # keep pretrained encoder
        prm.requires_grad = False
    params = [prm for prm in model.parameters() if prm.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ce = torch.nn.CrossEntropyLoss(ignore_index=255)
    losses, t0 = [], time.time()
    for step in range(1, steps + 1):
        model.train()
        model.encoder.eval()                             # frozen BatchNorm statistics
        x, y = batch(rgb, label, rng)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        logits = model(x)
        loss = ce(logits, y) + dice_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss))
        if step % 10 == 0 or step == steps:
            log(f'  step {step:4d}/{steps}  loss {np.mean(losses[-10:]):.3f}  ({time.time() - t0:.0f} s)')
    model.eval()
    return losses


# --------------------------- main --------------------------------------------
def main(steps=STEPS):
    import torch
    from matplotlib import pyplot as plt
    from evaluate_reference import Reference
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    run_dir = new_run_dir('finetune')
    lines = []

    def log(text=''):
        print(text, flush=True)
        lines.append(text)

    reference = Reference()
    log(f'Fine-tuning {md.REPO_ID}: train on {TRAIN} (reference annotation), test on {TEST}')
    log(f'  frozen encoder, {steps} steps x {BATCH} crops {CROP}px, lr {LR}, '
        f'enhancement {" + ".join(md.ENHANCE)}, upscaling x{md.SCALE}')
    rgb, label = training_data(reference)

    def evaluate(model, tag, save):
        res = {}
        for name in [TRAIN, TEST]:
            image, roi = load(name)
            enhanced, _ = enhancement.enhance(image, md.ENHANCE, roi)
            prob = md.predict_probability(model, enhanced, 'cpu', md.SCALE)
            polys, mask, final = md.buildings_from_probability(prob, roi, image.shape[1])
            role = 'TRAIN photo' if name == TRAIN else 'TEST photo '
            res[name] = len(polys)
            log(f'  {tag:11s} {name:8s} ({role}): {len(polys):2d} buildings  ' +
                reference.compare(name, image, roi, final, run_dir if save else None))
            if save:
                cv2.imwrite(os.path.join(run_dir, f'{name}_buildings_finetuned.png'),
                            draw_buildings(image, polys, roi))
                cv2.imwrite(os.path.join(run_dir, f'{name}_probability_finetuned.png'), md.heatmap(image, prob))
        return res

    model = md.load_model('cpu')
    log('\nBefore fine-tuning:')
    evaluate(model, 'pretrained', False)

    log('\nTraining:')
    losses = train(model, rgb, label, steps, log, rng)
    torch.save(model.state_dict(), os.path.join(run_dir, 'model_finetuned.pth'))

    log('\nAfter fine-tuning:')
    counts = evaluate(model, 'fine-tuned', True)
    log(f'\nBuilding count: ' + ', '.join(f'{n} = {c}' for n, c in counts.items()) + '  (reference: 31)')
    log(f'Weights: {os.path.abspath(os.path.join(run_dir, "model_finetuned.pth"))}')

    fig = plt.figure(figsize=(8, 4))
    plt.plot(losses, alpha=0.3, label='loss')
    plt.plot(np.convolve(losses, np.ones(10) / 10, 'valid'), label='moving average (10)')
    plt.xlabel('step'), plt.ylabel('cross-entropy + Dice'), plt.title('Fine-tuning loss')
    plt.legend(), plt.tight_layout()
    fig.savefig(os.path.join(run_dir, 'loss.png'), dpi=100)
    with open(os.path.join(run_dir, 'report.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Results saved to {os.path.abspath(run_dir)}')


if __name__ == '__main__':
    args = sys.argv[1:]
    main(int(args[args.index('--steps') + 1]) if '--steps' in args else STEPS)
