
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "") # forces the CPU, must be set before importing torch

import argparse
import json

import numpy as np
import cv2
from scipy.stats import spearmanr, kendalltau

from src.registry import AVAILABLE_MODELS, compute_saliency


# YOLO model not loaded, loaded only when needed
_yolo_model = None


# loads YOLOv8 nano only once and reuses it for all the images
def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt") # nano version: the lightest
    return _yolo_model


# detects the objects with YOLOv8 and returns a list of dicts {box: (x1, y1, x2, y2), label, conf}
def detect_objects(image_bgr):
    model = _load_yolo()
    results = model(image_bgr, verbose=False)[0] # a single image: first result
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int) # corners of the box in pixels
        label = results.names[int(box.cls[0])] # class name (e.g. "person")
        conf = float(box.conf[0]) # detection confidence
        boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": label, "conf": conf})
    return boxes


MAX_SALIENCY_SIDE = 800 # maximum long side of the image given to the saliency models


# shrinks the image if its long side exceeds max_side 
def resize_for_saliency(image_bgr, max_side=MAX_SALIENCY_SIDE):
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image_bgr # already small enough
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# computes the saliency on the shrunk image and brings the map back to the original resolution,
# so it can be indexed with the YOLO boxes, which are in the coordinates of the original image
def compute_saliency_at_full_resolution(model_name, image_bgr):
    h, w = image_bgr.shape[:2]
    small = resize_for_saliency(image_bgr)
    saliency_small = compute_saliency(model_name, small)
    if saliency_small.shape[:2] == (h, w):
        return saliency_small # image not shrunk, nothing to enlarge
    return cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_LINEAR)


# score of a box = mean of the saliency map inside it
def score_box(saliency_map, box):
    x1, y1, x2, y2 = box
    h, w = saliency_map.shape
    # keeps the box inside the borders of the map
    x1c, x2c = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
    y1c, y2c = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))
    if x2c <= x1c or y2c <= y1c:
        return 0.0 # empty box after clipping
    return float(saliency_map[y1c:y2c, x1c:x2c].mean())


# assigns to each box the order of the first click that falls inside it,
# boxes never clicked stay None and are excluded from the comparison
def ground_truth_rank(boxes, clicks):
    ordered_clicks = sorted(clicks, key=lambda c: c["order"]) # from the most to the least important click
    ranks = [None] * len(boxes)
    for click in ordered_clicks:
        for i, b in enumerate(boxes):
            if ranks[i] is not None:
                continue # box already assigned to a previous click
            x1, y1, x2, y2 = b["box"]
            if x1 <= click["x"] <= x2 and y1 <= click["y"] <= y2:
                ranks[i] = click["order"] # a click can assign more than one box if they overlap
    return ranks


# Spearman and Kendall tau-b between the model score and the click order, only on the clicked boxes;
# rank 1 = first click = most important, so -rank is used to have the same direction as the score
def compare_rankings(model_scores, gt_ranks):
    pairs = [(s, r) for s, r in zip(model_scores, gt_ranks) if r is not None] # only clicked boxes
    if len(pairs) < 2:
        return float("nan"), float("nan") # with fewer than 2 boxes there is no order to compare
    scores, ranks = zip(*pairs)
    rho, _ = spearmanr(scores, [-r for r in ranks]) # correlation between the positions in the ranking
    tau, _ = kendalltau(scores, [-r for r in ranks]) # fraction of pairs in the same order, tau-b handles ties
    return float(rho), float(tau)


# for each image detects the objects, builds the ground truth and compares the ranking of each model,
# returns {model: list of (rho, tau), one pair per image}
def evaluate_ranking(images_dir, clicks_path):
    with open(clicks_path) as f:
        all_clicks = json.load(f) # {image name: list of clicks}

    per_model_scores = {name: [] for name in AVAILABLE_MODELS}

    for name, clicks in all_clicks.items():
        path = os.path.join(images_dir, name)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"  ATTENZIONE: immagine non trovata: {name}, saltata.")
            continue

        # at least 2 objects are needed to talk about a ranking
        boxes = detect_objects(image_bgr)
        if len(boxes) < 2:
            print(f"  {name}: meno di 2 oggetti rilevati, saltata dal confronto ranking.")
            continue

        # and at least 2 of them must have been clicked
        gt_ranks = ground_truth_rank(boxes, clicks)
        n_clicked = sum(r is not None for r in gt_ranks)
        if n_clicked < 2:
            print(f"  {name}: meno di 2 box cliccate, saltata dal confronto ranking.")
            continue

        for model_name in AVAILABLE_MODELS:
            try:
                saliency = compute_saliency_at_full_resolution(model_name, image_bgr)
            except Exception as error:
                print(f"  {model_name} fallito su {name}: {error}") # a failing model does not block the others
                continue
            scores = [score_box(saliency, b["box"]) for b in boxes] # score of each box according to the model
            rho, tau = compare_rankings(scores, gt_ranks)
            per_model_scores[model_name].append((rho, tau))

        print(f"  {name}: {len(boxes)} oggetti rilevati, ranking confrontato su {n_clicked} box cliccate")

    return per_model_scores


# mean of Spearman and Kendall over all the images, for each model (NaNs are ignored)
def summarize(per_model_scores):
    summary = []
    for model_name, values in per_model_scores.items():
        if not values:
            continue # model with no evaluated images
        rhos = np.array([v[0] for v in values], dtype=np.float64)
        taus = np.array([v[1] for v in values], dtype=np.float64)
        rhos_valid = rhos[~np.isnan(rhos)] # discards the undefined cases (e.g. all scores equal)
        taus_valid = taus[~np.isnan(taus)]
        summary.append({
            "model": model_name,
            "n_images": len(values),
            "Spearman": round(float(rhos_valid.mean()), 4) if rhos_valid.size else float("nan"),
            "Kendall": round(float(taus_valid.mean()), 4) if taus_valid.size else float("nan"),
        })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Valuta il ranking oggetti guidato dalla saliency, confrontato con l'ordine dei click."
    )
    parser.add_argument("--images", default="img") # folder with the images
    parser.add_argument("--clicks", default="clicks.json") # file with the clicks saved by collect.py
    args = parser.parse_args()

    print(f"Valutazione ranking su '{args.images}' con ground truth da '{args.clicks}'\n")
    per_model = evaluate_ranking(args.images, args.clicks)
    summary = summarize(per_model)

    # final table: one row per model with the number of images and the mean of the two correlations
    header = f"\n{'Modello':<28}{'N img':>8}{'Spearman':>12}{'Kendall':>12}"
    print(header)
    print("-" * len(header.strip()))
    for e in summary:
        print(f"{e['model']:<28}{e['n_images']:>8}{e['Spearman']:>12.4f}{e['Kendall']:>12.4f}")
