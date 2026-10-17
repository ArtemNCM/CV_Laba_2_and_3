# ----------------  Lab 3: фільтрація та покращення якості зображень  ---------------

'''
Image quality improvement operations used before building detection.

Filtering (lecture examples: filtr_im.py, filtr_im_PIL.py)
    blur        - averaging, cv2.filter2D with a 5x5 kernel of 1/25
    gaussian    - cv2.GaussianBlur 5x5
    median      - cv2.medianBlur 5
    bilateral   - cv2.bilateralFilter 9, 75, 75 (smooths, keeps edges)
    sharpen     - PIL ImageEnhance.Sharpness, factor 2

Colour correction
    gray_world  - white balance: every channel gets the same mean (removes colour cast)
    dehaze      - dark channel prior (He et al.): removes haze / low-contrast veil
    gamma       - gamma correction of brightness

Brightness histogram correction (lecture examples: im quality enhanc.py)
    equalize    - histogram equalization of Lab-L through the cumulative histogram (CDF)
    clahe       - contrast limited adaptive histogram equalization of Lab-L (per tile)

Only the brightness channel L is changed by equalize / clahe / gamma, so roof colours
(Lab a, b) stay the same.

A chain is a list of names, e.g. ['gray_world', 'clahe', 'bilateral'].
'''

import cv2
import numpy as np
from PIL import Image, ImageEnhance


# --------------------------- filtering ---------------------------------------
def blur(img, roi=None):
    kernel = np.ones((5, 5), np.float32) / 25                 # averaging, as in filtr_im.py
    return cv2.filter2D(img, -1, kernel)


def gaussian(img, roi=None):
    return cv2.GaussianBlur(img, (5, 5), 0)


def median(img, roi=None):
    return cv2.medianBlur(img, 5)


def bilateral(img, roi=None):
    return cv2.bilateralFilter(img, 9, 75, 75)


def sharpen(img, roi=None):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Sharpness(pil).enhance(2.0)            # as in filtr_im.py
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# --------------------------- colour correction -------------------------------
def gray_world(img, roi=None):
    # white balance: scale B, G, R so that their means inside the ROI are equal
    f = img.astype(np.float32)
    m = roi if roi is not None else np.ones(img.shape[:2], bool)
    means = f[m].mean(axis=0)
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


def dehaze(img, roi=None, omega=0.85, patch=15, t_min=0.2):
    # dark channel prior: in haze-free outdoor images some channel is close to 0
    # in every small patch; haze lifts it -> estimate transmission t and invert
    #     I = J * t + A * (1 - t)   =>   J = (I - A) / t + A
    f = img.astype(np.float32) / 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(f.min(axis=2), kernel)
    n = max(1, dark.size // 1000)                              # brightest 0.1% of dark channel
    idx = np.argsort(dark.ravel())[-n:]
    A = f.reshape(-1, 3)[idx].mean(axis=0)                     # atmospheric light
    t = 1 - omega * cv2.erode((f / A).min(axis=2), kernel)
    t = cv2.GaussianBlur(t, (0, 0), patch)                     # smooth transmission map
    t = np.maximum(t, t_min)[..., None]
    J = (f - A) / t + A
    return np.clip(J * 255, 0, 255).astype(np.uint8)


def _on_lightness(img, func):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = func(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma(img, roi=None, g=0.8):
    lut = np.clip(((np.arange(256) / 255) ** g) * 255, 0, 255).astype(np.uint8)
    return _on_lightness(img, lambda L: lut[L])


# --------------------------- histogram correction ----------------------------
def equalize_cdf(channel, roi=None):
    # histogram equalization through the cumulative histogram, as in im quality enhanc.py;
    # the CDF is computed inside the ROI only (black area outside would distort it)
    values = channel[roi] if roi is not None else channel.ravel()
    hist, _ = np.histogram(values, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    lut = np.ma.filled(cdf_m, 0).astype(np.uint8)
    return lut[channel]


def equalize(img, roi=None):
    return _on_lightness(img, lambda L: equalize_cdf(L, roi))


def clahe(img, roi=None):
    op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _on_lightness(img, op.apply)


OPERATIONS = dict(blur=blur, gaussian=gaussian, median=median, bilateral=bilateral,
                  sharpen=sharpen, gray_world=gray_world, dehaze=dehaze, gamma=gamma,
                  equalize=equalize, clahe=clahe)


def enhance(img, chain, roi=None):
    # apply the operations one after another; returns the final image and every step
    steps = [('original', img)]
    for name in chain:
        img = OPERATIONS[name](img, roi)
        steps.append((name, img))
    return img, steps


# --------------------------- quality analysis --------------------------------
def quality(img, roi=None):
    # simple numbers that describe image quality inside the ROI
    m = roi if roi is not None else np.ones(img.shape[:2], bool)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F)[m].var()          # focus measure
    hist = np.bincount(L[m], minlength=256) / m.sum()
    entropy = -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()
    return dict(brightness=float(L[m].mean()), contrast=float(L[m].std()),
                entropy=float(entropy), sharpness=float(sharp))


def histogram_figure(title, steps, roi=None):
    # images of every step with their brightness histograms and CDF (lecture style)
    from matplotlib import pyplot as plt
    n = len(steps)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False)
    m = roi if roi is not None else np.ones(steps[0][1].shape[:2], bool)
    for i, (name, im) in enumerate(steps):
        axes[0, i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(name), axes[0, i].set_xticks([]), axes[0, i].set_yticks([])
        L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0][m]
        hist, _ = np.histogram(L, 256, [0, 256])
        cdf = hist.cumsum()
        axes[1, i].hist(L, 256, [0, 256], color='r')
        axes[1, i].plot(cdf * hist.max() / cdf.max(), color='b')
        axes[1, i].set_xlim([0, 256]), axes[1, i].set_yticks([])
        q = quality(im, roi)
        axes[1, i].set_title(f"contrast {q['contrast']:.0f}, entropy {q['entropy']:.2f}", fontsize=9)
    axes[1, 0].legend(('cdf', 'histogram'), loc='upper left', fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
