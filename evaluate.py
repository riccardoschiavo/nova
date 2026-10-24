import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "") # forces the CPU, must be set before importing torch

import argparse
import json

import numpy as np
import cv2

from src.registry import AVAILABLE_MODELS, compute_saliency


EPS = 1e-12 # threshold below which a standard deviation or a range is considered zero
SIGMA_FRAC = 0.05 # same value used in collect.py: ~1 degree of visual angle
MAX_SALIENCY_SIDE = 800 # maximum long side of the image given to the saliency models


# creates the heatmap by summing a gaussian centered on each click,
# same logic as collect.py and osie_ground_truth.py (used for CC)
def clicks_to_heatmap(clicks, height, width):
    heatmap = np.zeros((height, width), dtype=np.float32)
    if not clicks:
        return heatmap # no clicks: empty map
    sigma = SIGMA_FRAC * width # sigma proportional to the image width
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32) # coordinates of every pixel
    for c in clicks:
        d2 = (x_grid - c["x"]) ** 2 + (y_grid - c["y"]) ** 2 # squared distance from the click
        heatmap += np.exp(-d2 / (2.0 * sigma ** 2)) # gaussian centered on the click
    # normalizes to [0, 1] by dividing by the maximum
    m = heatmap.max()
    return heatmap / m if m > 0 else heatmap


# creates the binary click map: 1 on the clicked pixels, 0 elsewhere (used for NSS and AUC-Judd)
def clicks_to_fixation_map(clicks, height, width):
    fmap = np.zeros((height, width), dtype=np.float32)
    for c in clicks:
        x = int(np.clip(c["x"], 0, width - 1)) # keeps the click inside the image
        y = int(np.clip(c["y"], 0, height - 1))
        fmap[y, x] = 1.0
    return fmap


# shrinks the image if its long side exceeds max_side
def resize_for_saliency(image_bgr, max_side=MAX_SALIENCY_SIDE):
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image_bgr # already small enough
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# computes the saliency on the shrunk image and brings the map back to the original resolution,
# so it can be compared pixel by pixel with the ground truth built from the full resolution clicks
def compute_saliency_at_full_resolution(model_name, image_bgr):
    h, w = image_bgr.shape[:2]
    small = resize_for_saliency(image_bgr)
    saliency_small = compute_saliency(model_name, small)
    if saliency_small.shape[:2] == (h, w):
        return saliency_small # image not shrunk, nothing to enlarge
    return cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_LINEAR)


# brings the map to mean 0 and standard deviation 1, so maps on different scales are comparable
def _standardize(m):
    m = m.astype(np.float64)
    std = m.std()
    return (m - m.mean()) / std if std > EPS else np.zeros_like(m) # constant map: all zeros


# CC: Pearson correlation between saliency and heatmap,
# 1 = same trend, 0 = no relation, -1 = opposite trend
def cc(saliency, ground_truth):
    return float((_standardize(saliency) * _standardize(ground_truth)).mean())


# NSS: mean of the standardized saliency computed only on the clicked pixels,
# 0 = like chance, positive = the clicks fall on areas more salient than average
def nss(saliency, fixation_map):
    fixations = fixation_map > 0
    if not fixations.any():
        return float("nan") # no clicks: metric not defined
    return float(_standardize(saliency)[fixations].mean())


# AUC-Judd: area under the ROC curve, the map acts as a classifier between clicked and non clicked pixels,
# 0.5 = like chance, 1 = perfect separation
def auc_judd(saliency, fixation_map):
    fixations = fixation_map > 0
    if not fixations.any():
        return float("nan") # no clicks
        
    # normalizes the saliency to [0, 1]
    s = saliency.astype(np.float64)
    lo, hi = s.min(), s.max()
    if hi - lo < EPS:
        return 0.5 # constant map: distinguishes nothing, like chance
    s = (s - lo) / (hi - lo)

    fix_vals = np.sort(s[fixations])[::-1] # saliency on the clicks, from highest to lowest: these are the thresholds
    all_vals = np.sort(s.ravel()) # saliency of all the pixels, sorted for searchsorted
    n_fix, n_pix = fix_vals.size, all_vals.size
    n_non_fix = n_pix - n_fix
    if n_non_fix <= 0:
        return 0.5 # all pixels are clicks: there are no negatives

    # one point of the ROC curve for each threshold, starting from (0, 0)
    tp, fp = [0.0], [0.0]
    for i, t in enumerate(fix_vals):
        above = n_pix - np.searchsorted(all_vals, t, side="left") # pixels with saliency >= threshold
        tp.append((i + 1) / n_fix) # fraction of clicks above the threshold
        fp.append((above - (i + 1)) / n_non_fix) # fraction of non clicked pixels above the threshold
    tp.append(1.0) # the curve ends at (1, 1)
    fp.append(1.0)
    tp, fp = np.array(tp), np.array(fp)
    return float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2.0)) # area with the trapezoidal rule


METRICS = {"CC": cc, "NSS": nss, "AUC-Judd": auc_judd}
NEEDS_FIXATIONS = {"CC": False, "NSS": True, "AUC-Judd": True} # True = uses the binary map, False = the heatmap


# computes the three metrics for each (image, model) pair and returns one row per pair
def evaluate(images_dir, clicks_path):
    with open(clicks_path) as f:
        all_clicks = json.load(f) # {image name: list of clicks}

    rows = []
    for name, clicks in all_clicks.items():
        path = os.path.join(images_dir, name)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"  ATTENZIONE: immagine non trovata: {name}, saltata.")
            continue

        # ground truth built only once per image, valid for all the models
        h, w = image_bgr.shape[:2]
        gt_heatmap = clicks_to_heatmap(clicks, h, w)
        gt_fixations = clicks_to_fixation_map(clicks, h, w)

        for model_name in AVAILABLE_MODELS:
            try:
                saliency = compute_saliency_at_full_resolution(model_name, image_bgr)
            except Exception as error:
                print(f"  {model_name} fallito su {name}: {error}") # a failing model does not block the others
                continue

            row = {"image": name, "model": model_name}
            for metric_name, fn in METRICS.items():
                reference = gt_fixations if NEEDS_FIXATIONS[metric_name] else gt_heatmap # right ground truth for the metric
                row[metric_name] = round(fn(saliency, reference), 4)
            rows.append(row)

        print(f"  {name}: valutata su {len(AVAILABLE_MODELS)} modelli")

    return rows


# mean of each metric over all the images, for each model (NaNs are ignored)
def summarize(rows):
    models = sorted(set(r["model"] for r in rows))
    summary = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        entry = {"model": model, "n_images": len(model_rows)}
        for metric_name in METRICS:
            values = np.array([r[metric_name] for r in model_rows], dtype=np.float64)
            valid = values[~np.isnan(values)] # discards the images without clicks
            entry[metric_name] = round(float(valid.mean()), 4) if valid.size else float("nan")
        summary.append(entry)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuta i modelli NOVA contro i click raccolti.")
    parser.add_argument("--images", default="img") # folder with the images
    parser.add_argument("--clicks", default="clicks.json") # file with the clicks saved by collect.py
    args = parser.parse_args()

    print(f"Valutazione su '{args.images}' con ground truth da '{args.clicks}'\n")
    rows = evaluate(args.images, args.clicks)
    summary = summarize(rows)

    # final table: one row per model with the mean of the three metrics
    header = f"\n{'Modello':<24}{'CC':>10}{'NSS':>10}{'AUC-Judd':>10}"
    print(header)
    print("-" * len(header.strip()))
    for e in summary:
        print(f"{e['model']:<24}{e['CC']:>10.4f}{e['NSS']:>10.4f}{e['AUC-Judd']:>10.4f}")
