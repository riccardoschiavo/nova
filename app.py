"""NOVA — unified demo: every feature of the project in a single Gradio app.

Tabs:
  1. Saliency         baselines and deep models side by side on img/ (or an uploaded image)
  2. Ranking oggetti  YOLOv8 boxes ranked by mean saliency, compared with the click order
  3. Raccolta click   the collect.py annotation tool (writes clicks.json)
  4. Valutazione      experiment 1 (CC, NSS, AUC-Judd) and experiment 2 (Spearman, Kendall)
  5. Fine-tuning      OSIE ground-truth preview and the fine-tuning scripts

Run with:  python app.py   (then open http://127.0.0.1:7860)
"""
import hashlib
import importlib
import importlib.util
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import multiprocessing
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

import collect
import evaluate
import rank_objects
from osie_ground_truth import load_fixations, points_to_heatmap
from src.baselines.center_prior import center_prior_saliency
from src.baselines.itti_koch import itti_koch_saliency
from src.baselines.spectral_residual import spectral_residual_saliency
from src.registry import compute_saliency

DEEP_SPECS = {
    "MSI-Net": ("src.models.msi_net", "msinet_saliency",
                ("tensorflow", "tensorflow_hub"),
                "pip install tensorflow tensorflow_hub"),
    "DeepGaze IIE": ("src.models.deepgaze", "deepgaze_saliency",
                     ("torch", "deepgaze_pytorch"),
                     "pip install torch git+https://github.com/matthias-k/DeepGaze.git"),
    "DeepGaze IIE (fine-tuned)": ("src.models.deepgaze", "deepgaze_finetuned_saliency",
                                  ("torch", "deepgaze_pytorch"),
                                  "pip install torch git+https://github.com/matthias-k/DeepGaze.git"),
}
DEEP_NAMES = list(DEEP_SPECS)
# names as in src/registry.py, used by the ranking and the evaluation
BASELINE_NAMES = ["Center Prior", "Itti-Koch", "Spectral Residual"]
ALL_MODELS = BASELINE_NAMES + DEEP_NAMES


def deep_missing(name):
    return [m for m in DEEP_SPECS[name][2] if importlib.util.find_spec(m) is None]

ROOT = Path(__file__).parent
IMG_DIR = ROOT / "img"
IMAGE_PATHS = sorted(IMG_DIR.glob("*.jpeg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
CLICKS_PATH = ROOT / "clicks.json"
FINETUNED_WEIGHTS = ROOT / "deepgaze_finetuned.pt"
OSIE_DIR = ROOT / "predicting-human-gaze-beyond-pixels"

# Downscale large images before processing so the UI stays responsive
MAX_SIDE = 800


def load_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


@lru_cache(maxsize=256)
def original_shape(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR).shape[:2]


def to_heatmap(saliency):
    # saliency in [0, 1] -> RGB heatmap
    colored = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def to_overlay(image_bgr, saliency, alpha):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, to_heatmap(saliency), alpha, 0)


# ---------------------------------------------------------------- clicks

def load_clicks():
    if not CLICKS_PATH.exists():
        return {}
    with open(CLICKS_PATH) as f:
        return json.load(f)


def clicks_for(path):
    # clicks exist only for the images in img/ (an upload may share a file name by chance)
    if path is None or Path(path).parent.resolve() != IMG_DIR.resolve():
        return None
    return load_clicks().get(Path(path).name)


def draw_scaled_clicks(rgb, clicks, scale):
    # clicks are in original-image pixels, rgb is the downscaled image
    ordered = sorted(clicks, key=lambda c: c["order"])
    return collect.draw_clicks(rgb, [(int(c["x"] * scale), int(c["y"] * scale)) for c in ordered])


# ---------------------------------------------------------------- saliency maps

# Deep models run in a short-lived child process: the process exits after each job,
# so TensorFlow (MSI-Net) and PyTorch (DeepGaze) are never in RAM together.
# Maps are cached on disk so each (model, image) pair is computed only once.
CACHE_DIR = ROOT / ".cache" / "saliency"
_deep_pool = None


def _deep_worker(module, function, images):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    fn = getattr(importlib.import_module(module), function)
    return [fn(image) for image in images]


def _deep_executor():
    global _deep_pool
    if _deep_pool is None:
        _deep_pool = ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1)
    return _deep_pool


def _cache_file(name, path):
    digest = hashlib.sha1(Path(path).read_bytes())
    if name == "DeepGaze IIE (fine-tuned)" and FINETUNED_WEIGHTS.exists():
        # retrained weights get new cache entries instead of showing the old maps
        digest.update(str(FINETUNED_WEIGHTS.stat().st_mtime_ns).encode())
    slug = "".join(c for c in name.lower() if c.isalnum())
    return CACHE_DIR / f"{slug}_{digest.hexdigest()[:16]}_{MAX_SIDE}.npy"


def deep_saliency_many(name, paths):
    # one child process for all the uncached images, so the model is loaded only once
    global _deep_pool
    missing = deep_missing(name)
    if missing:
        raise RuntimeError(f"non disponibile, manca {', '.join(missing)} (`{DEEP_SPECS[name][3]}`)")
    files = [_cache_file(name, p) for p in paths]
    todo = [i for i, f in enumerate(files) if not f.exists()]
    if todo:
        module, function = DEEP_SPECS[name][:2]
        images = [load_bgr(paths[i]) for i in todo]
        try:
            maps = _deep_executor().submit(_deep_worker, module, function, images).result()
        except BrokenProcessPool:
            _deep_pool.shutdown(wait=False)
            _deep_pool = None
            raise RuntimeError(f"il processo di {name} è terminato in modo anomalo (memoria esaurita?)")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for i, saliency in zip(todo, maps):
            np.save(files[i], saliency)
    return [np.load(f) for f in files]


def deep_saliency(name, path):
    return deep_saliency_many(name, [path])[0]


@lru_cache(maxsize=128)
def _baseline_saliency(name, path):
    return compute_saliency(name, load_bgr(path))


def model_saliency(name, path):
    # map at the demo resolution (longer side MAX_SIDE); baselines use their default parameters
    if name in DEEP_SPECS:
        return deep_saliency(name, str(path))
    return _baseline_saliency(name, str(path))


def full_resolution(saliency, path):
    # back to the original resolution, where clicks and YOLO boxes live
    h, w = original_shape(str(path))
    if saliency.shape[:2] == (h, w):
        return saliency
    return cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- tab 1: saliency

def make_renderer(image, alpha, view):
    return {
        "Heatmap": lambda m: to_heatmap(m),
        "Overlay": lambda m: to_overlay(image, m, alpha),
        "Grayscale": lambda m: (m * 255).astype(np.uint8),
    }[view]


def run(path, sigma_frac, sr_sigma, sr_size, alpha, view, show_clicks):
    if path is None:
        return [None] * 4 + [""]
    image = load_bgr(path)

    timings = {}
    maps = {}

    t = time.perf_counter()
    maps["Center prior"] = center_prior_saliency(image.shape[:2], sigma_frac=sigma_frac)
    timings["Center prior"] = time.perf_counter() - t

    t = time.perf_counter()
    maps["Itti-Koch"] = itti_koch_saliency(image)
    timings["Itti-Koch"] = time.perf_counter() - t

    t = time.perf_counter()
    maps["Spectral residual"] = spectral_residual_saliency(
        image, sigma=sr_sigma, target_size=(int(sr_size), int(sr_size)))
    timings["Spectral residual"] = time.perf_counter() - t

    render = make_renderer(image, alpha, view)
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    clicks = clicks_for(path)
    if show_clicks and clicks:
        original = draw_scaled_clicks(original, clicks, image.shape[1] / original_shape(str(path))[1])
    click_note = f"{len(clicks)} click annotati" if clicks else "nessun click annotato"
    info = (f"**{Path(path).name}** — {image.shape[1]}×{image.shape[0]} px — {click_note}  \n"
            + " · ".join(f"{name}: {sec * 1000:.0f} ms" for name, sec in timings.items()))
    return (original,
            render(maps["Center prior"]),
            render(maps["Itti-Koch"]),
            render(maps["Spectral residual"]),
            info)


def run_deep(path, alpha, view, deep_selected):
    if path is None:
        return [None] * len(DEEP_NAMES) + [""]
    image = load_bgr(path)
    render = make_renderer(image, alpha, view)

    images = []
    notes = []
    for name in DEEP_NAMES:
        if name not in deep_selected:
            images.append(None)
            notes.append(f"{name}: disattivato")
            continue
        t = time.perf_counter()
        try:
            saliency = deep_saliency(name, str(path))
        except Exception as exc:
            images.append(None)
            notes.append(f"{name}: errore ({exc})")
            continue
        images.append(render(saliency))
        notes.append(f"{name}: {(time.perf_counter() - t) * 1000:.0f} ms")
    return (*images, "  \n".join(notes))


# ---------------------------------------------------------------- tab 2: object ranking

def yolo_missing():
    return importlib.util.find_spec("ultralytics") is None


@lru_cache(maxsize=64)
def detect_cached(path):
    return rank_objects.detect_objects(cv2.imread(path, cv2.IMREAD_COLOR))


def _rank_color(rank, n):
    # JET colour: red = most salient box, blue = least salient
    value = 255 - int(255 * (rank - 1) / max(n - 1, 1))
    bgr = cv2.applyColorMap(np.uint8([[value]]), cv2.COLORMAP_JET)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def _draw_label(rgb, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    cv2.rectangle(rgb, (x, y - th - 6), (x + tw + 6, y), (0, 0, 0), -1)
    cv2.putText(rgb, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def run_ranking(path, model_name, alpha):
    if path is None:
        return None, None, "Seleziona un'immagine nella scheda **Saliency**."
    if yolo_missing():
        return None, None, "YOLOv8 non disponibile: `pip install ultralytics`."
    name = Path(path).name
    boxes = detect_cached(str(path))
    try:
        saliency = model_saliency(model_name, path)
    except Exception as exc:
        return None, None, f"**{name}** — {model_name}: errore ({exc})"

    scores = [rank_objects.score_box(full_resolution(saliency, path), b["box"]) for b in boxes]
    clicks = clicks_for(path)
    gt_ranks = rank_objects.ground_truth_rank(boxes, clicks) if clicks else [None] * len(boxes)
    order = np.argsort(scores)[::-1]
    pred_ranks = np.empty(len(boxes), dtype=int)
    pred_ranks[order] = np.arange(1, len(boxes) + 1)

    image = load_bgr(path)
    scale = image.shape[1] / original_shape(str(path))[1]
    annotated = to_overlay(image, saliency, alpha)
    for i in order[::-1]:  # least salient first, so the top boxes are drawn on top
        x1, y1, x2, y2 = (int(round(v * scale)) for v in boxes[i]["box"])
        color = _rank_color(pred_ranks[i], len(boxes))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        _draw_label(annotated, f"#{pred_ranks[i]} {boxes[i]['label']}", x1, y1, color)
    if clicks:
        annotated = draw_scaled_clicks(annotated, clicks, scale)

    table = pd.DataFrame([{
        "Rank modello": int(pred_ranks[i]),
        "Oggetto": boxes[i]["label"],
        "Confidenza YOLO": round(boxes[i]["conf"], 3),
        "Saliency media": round(scores[i], 4),
        "Ordine click": gt_ranks[i] if gt_ranks[i] is not None else "—",
    } for i in order])

    info = f"**{name}** — {model_name} — {len(boxes)} oggetti rilevati da YOLOv8n.  \n"
    n_clicked = sum(r is not None for r in gt_ranks)
    if not clicks:
        info += "Nessun click per questa immagine: annotala nella scheda **Raccolta click** per il confronto."
    elif len(boxes) < 2 or n_clicked < 2:
        info += f"Box cliccate: {n_clicked} — servono almeno 2 oggetti cliccati per Spearman/Kendall."
    else:
        rho, tau = rank_objects.compare_rankings(scores, gt_ranks)
        info += f"Confronto su {n_clicked} box cliccate: **Spearman ρ = {rho:.4f}**, **Kendall τ-b = {tau:.4f}**"
    return annotated, table, info


# ---------------------------------------------------------------- tab 4: evaluation

def run_evaluation(models):
    # same procedure as evaluate.py and rank_objects.py, but with the cached maps of the demo
    def emit(pixel=None, ranking=None, per_image=None):
        return "  \n".join(log), pixel, ranking, per_image

    log = []
    clicks = load_clicks()
    names = [n for n in clicks if (IMG_DIR / n).exists()]
    if not names or not models:
        log.append("Nessuna immagine annotata in `clicks.json`." if not names else "Seleziona almeno un modello.")
        yield emit()
        return
    paths = {n: str(IMG_DIR / n) for n in names}
    models = [m for m in ALL_MODELS if m in models]
    log.append(f"Valutazione di {len(models)} modelli su {len(names)} immagini annotate.")
    yield emit()

    # deep maps first, one child process per model
    failed = set()
    for model in models:
        if model in DEEP_SPECS:
            log.append(f"{model}: calcolo delle mappe…")
            yield emit()
            t = time.perf_counter()
            try:
                deep_saliency_many(model, list(paths.values()))
                log[-1] = f"{model}: mappe pronte ({time.perf_counter() - t:.1f} s)"
            except Exception as exc:
                failed.add(model)
                log[-1] = f"{model}: errore ({exc}), escluso"
            yield emit()

    with_ranking = not yolo_missing()
    if not with_ranking:
        log.append("YOLOv8 non disponibile (`pip install ultralytics`): esperimento 2 saltato.")
    models = [m for m in models if m not in failed]
    rows = []
    per_model_ranking = {m: [] for m in models}
    for name in names:
        path = paths[name]
        h, w = original_shape(path)
        gt_heatmap = evaluate.clicks_to_heatmap(clicks[name], h, w)
        gt_fixations = evaluate.clicks_to_fixation_map(clicks[name], h, w)

        boxes, gt_ranks = [], []
        if with_ranking:
            boxes = detect_cached(path)
            gt_ranks = rank_objects.ground_truth_rank(boxes, clicks[name])
        use_ranking = len(boxes) >= 2 and sum(r is not None for r in gt_ranks) >= 2

        for model in models:
            saliency = full_resolution(model_saliency(model, path), path)
            row = {"image": name, "model": model}
            for metric, fn in evaluate.METRICS.items():
                reference = gt_fixations if evaluate.NEEDS_FIXATIONS[metric] else gt_heatmap
                row[metric] = round(fn(saliency, reference), 4)
            rows.append(row)
            if use_ranking:
                scores = [rank_objects.score_box(saliency, b["box"]) for b in boxes]
                per_model_ranking[model].append(rank_objects.compare_rankings(scores, gt_ranks))

        note = f"{len(boxes)} oggetti, usata nel ranking" if use_ranking else "esclusa dal ranking"
        log.append(f"{name}: valutata ({note})" if with_ranking else f"{name}: valutata")
        yield emit()

    pixel = pd.DataFrame(evaluate.summarize(rows))
    pixel = pixel.set_index("model").loc[models].reset_index()  # keep the model order of the app
    ranking = pd.DataFrame(rank_objects.summarize(per_model_ranking)) if with_ranking else None
    if ranking is not None and ranking.empty:
        log.append("Nessuna immagine con almeno 2 oggetti cliccati: ranking non calcolabile.")
        ranking = None
    log.append("**Fatto.**")
    yield emit(pixel, ranking, pd.DataFrame(rows))


# ---------------------------------------------------------------- tab 5: OSIE and fine-tuning

@lru_cache(maxsize=2)
def _osie_fixations(mat_path):
    return load_fixations(mat_path)


def osie_names(osie_dir):
    mat_path = Path(osie_dir) / "data" / "eye" / "fixations.mat"
    if not mat_path.exists():
        return None
    return list(_osie_fixations(str(mat_path)))


def run_osie_preview(osie_dir, index, model_name, alpha):
    names = osie_names(osie_dir)
    if names is None:
        return None, None, f"`data/eye/fixations.mat` non trovato in `{osie_dir}`."
    index = int(np.clip(index, 0, len(names) - 1))
    name = names[index]
    path = Path(osie_dir) / "data" / "stimuli" / name
    if not path.exists():
        return None, None, f"Immagine `{name}` non trovata."
    points = _osie_fixations(str(Path(osie_dir) / "data" / "eye" / "fixations.mat"))[name]

    image = load_bgr(path)
    h, w = original_shape(str(path))
    scale = image.shape[1] / w
    gt = points_to_heatmap(points, (h, w))
    gt_view = to_overlay(image, cv2.resize(gt, (image.shape[1], image.shape[0])), alpha)
    for x, y in points:
        cv2.circle(gt_view, (int(x * scale), int(y * scale)), 3, (255, 255, 255), -1)

    info = (f"**{name}** (indice {index} di {len(names)}) — {len(points)} fissazioni di tutti gli osservatori.  \n"
            "Le prime 200 immagini sono quelle usate di default per il fine-tuning.")
    if model_name == "(nessuno)":
        return gt_view, None, info
    try:
        saliency = model_saliency(model_name, path)
    except Exception as exc:
        return gt_view, None, info + f"  \n{model_name}: errore ({exc})"
    full = full_resolution(saliency, path)
    fixation_map = evaluate.clicks_to_fixation_map([{"x": x, "y": y} for x, y in points], h, w)
    scores = {"CC": evaluate.cc(full, gt), "NSS": evaluate.nss(full, fixation_map),
              "AUC-Judd": evaluate.auc_judd(full, fixation_map)}
    info += f"  \n{model_name} vs fissazioni OSIE: " + " · ".join(f"**{k}** {v:.4f}" for k, v in scores.items())
    return gt_view, to_overlay(image, saliency, alpha), info


# only DeepGaze IIE can be fine-tuned: the MSI-Net SavedModel exposes its weights as frozen
FINETUNE_SCRIPT = ROOT / "finetune_deepgaze.py"
_train_proc = None


def run_finetune(osie_dir, max_images, epochs, batch_size, learning_rate, pattern, inspect_only):
    global _train_proc
    if _train_proc is not None and _train_proc.poll() is None:
        yield "Un fine-tuning è già in corso."
        return
    if osie_names(osie_dir) is None:
        yield f"`data/eye/fixations.mat` non trovato in '{osie_dir}'. Clona prima il repository OSIE."
        return
    pattern = pattern.strip()
    if not pattern and not inspect_only:
        yield "Indica un unfreeze pattern (usa «Solo ispezione» per vedere i nomi dei parametri)."
        return

    cmd = [sys.executable, "-u", str(FINETUNE_SCRIPT), "--osie", str(osie_dir),
           "--max-images", str(int(max_images)), "--epochs", str(int(epochs)),
           "--batch-size", str(int(batch_size)), "--learning-rate", str(learning_rate)]
    if pattern:
        cmd += ["--unfreeze-pattern", pattern]
    if inspect_only:
        cmd.append("--inspect-only")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")

    log = "$ python " + shlex.join(cmd[2:]) + "\n\n"
    yield log
    _train_proc = subprocess.Popen(cmd, cwd=ROOT, env=env, text=True, bufsize=1,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for line in _train_proc.stdout:
            log += line
            yield log
        code = _train_proc.wait()
    finally:
        # also reached when the run is cancelled with the Stop button
        if _train_proc.poll() is None:
            _train_proc.terminate()
    log += f"\n[processo terminato con codice {code}]"
    if code == 0 and not inspect_only:
        log += "\nI nuovi pesi sono usati subito da «DeepGaze IIE (fine-tuned)» (la cache viene rigenerata)."
    yield log


def stop_finetune():
    if _train_proc is not None and _train_proc.poll() is None:
        _train_proc.terminate()


# ---------------------------------------------------------------- UI

with gr.Blocks(title="NOVA") as demo:
    gr.Markdown("# NOVA — Gaze-guided Salient Object Ranking")
    selected_path = gr.State(str(IMAGE_PATHS[0]) if IMAGE_PATHS else None)

    with gr.Tabs():
        with gr.Tab("1 · Saliency"):
            gr.Markdown("Seleziona un'immagine dalla galleria per confrontare i metodi. "
                        "L'immagine scelta è usata anche nella scheda **Ranking oggetti**.")
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    gallery = gr.Gallery(
                        value=[str(p) for p in IMAGE_PATHS], label="img/", columns=3,
                        height=420, allow_preview=False, object_fit="cover",
                    )
                    upload = gr.Image(label="…oppure carica un'immagine", type="filepath", height=160)

                    view = gr.Radio(["Heatmap", "Overlay", "Grayscale"], value="Overlay", label="Visualizzazione")
                    alpha = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Opacità overlay")
                    show_clicks = gr.Checkbox(value=True, label="Mostra i click (ground truth) sull'originale")

                    with gr.Accordion("Center prior", open=False):
                        sigma_frac = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="sigma_frac")
                    with gr.Accordion("Spectral residual", open=False):
                        sr_sigma = gr.Slider(0.5, 8.0, value=3.0, step=0.5, label="sigma (blur)")
                        sr_size = gr.Slider(32, 256, value=64, step=16, label="target_size (px)")

                with gr.Column(scale=3):
                    info = gr.Markdown()
                    with gr.Row():
                        out_original = gr.Image(label="Originale", interactive=False)
                        out_center = gr.Image(label="Center prior", interactive=False)
                    with gr.Row():
                        out_itti = gr.Image(label="Itti-Koch", interactive=False)
                        out_sr = gr.Image(label="Spectral residual", interactive=False)

                    deep_selected = gr.CheckboxGroup(
                        DEEP_NAMES, value=[n for n in DEEP_NAMES if not deep_missing(n)],
                        label="Modelli deep (il primo calcolo scarica i pesi e può richiedere decine di secondi)")
                    deep_info = gr.Markdown()
                    with gr.Row():
                        out_msinet = gr.Image(label="MSI-Net", interactive=False)
                        out_deepgaze = gr.Image(label="DeepGaze IIE", interactive=False)
                        out_deepgaze_ft = gr.Image(label="DeepGaze IIE (fine-tuned)", interactive=False)

        with gr.Tab("2 · Ranking oggetti") as ranking_tab:
            gr.Markdown("YOLOv8n rileva gli oggetti nell'immagine selezionata nella scheda **Saliency**; "
                        "ogni box riceve come punteggio la saliency media al suo interno. Se l'immagine è "
                        "annotata, il ranking è confrontato con l'ordine dei click (cerchi numerati).")
            with gr.Row():
                rank_model = gr.Dropdown(ALL_MODELS, value="DeepGaze IIE", label="Modello di saliency")
                rank_btn = gr.Button("Calcola ranking", variant="primary")
            rank_info = gr.Markdown()
            with gr.Row():
                rank_image = gr.Image(label="Box ordinate per saliency (rosso = più saliente)",
                                      interactive=False, scale=3)
                rank_table = gr.Dataframe(label="Oggetti", interactive=False, scale=2)

        with gr.Tab("3 · Raccolta click"):
            # a Blocks created inside another one is rendered into it automatically
            collect.build_interface(str(IMG_DIR), str(CLICKS_PATH))

        with gr.Tab("4 · Valutazione"):
            gr.Markdown(
                "Esegue i due esperimenti su tutte le immagini annotate in `clicks.json`:  \n"
                "**1. Pixel level** — CC, NSS e AUC-Judd tra la mappa e i click.  \n"
                "**2. Ranking oggetti** — Spearman ρ e Kendall τ-b tra il ranking delle box YOLO "
                "e l'ordine dei click (solo immagini con almeno 2 oggetti cliccati).  \n"
                "Le mappe deep sono condivise con la scheda Saliency tramite la cache su disco.")
            eval_models = gr.CheckboxGroup(
                ALL_MODELS, value=[m for m in ALL_MODELS if m not in DEEP_SPECS or not deep_missing(m)],
                label="Modelli")
            eval_btn = gr.Button("Esegui valutazione", variant="primary")
            eval_log = gr.Markdown()
            eval_pixel = gr.Dataframe(label="Esperimento 1 — media per modello", interactive=False)
            eval_ranking = gr.Dataframe(label="Esperimento 2 — media per modello", interactive=False)
            with gr.Accordion("Esperimento 1 — dettaglio per immagine", open=False):
                eval_rows = gr.Dataframe(interactive=False)

        with gr.Tab("5 · Fine-tuning"):
            osie_dir = gr.Textbox(str(OSIE_DIR), label="Cartella del repository OSIE")
            with gr.Accordion("Ground truth OSIE", open=True):
                gr.Markdown("Heatmap delle fissazioni costruita da `osie_ground_truth.py` "
                            "(σ = 0.05 × larghezza), opzionalmente confrontata con un modello.")
                with gr.Row():
                    osie_index = gr.Number(0, precision=0, minimum=0, label="Indice immagine")
                    osie_model = gr.Dropdown(["(nessuno)"] + ALL_MODELS, value="(nessuno)",
                                             label="Confronta con")
                    osie_btn = gr.Button("Mostra", variant="primary")
                osie_info = gr.Markdown()
                with gr.Row():
                    osie_gt = gr.Image(label="Fissazioni + heatmap", interactive=False)
                    osie_pred = gr.Image(label="Mappa del modello", interactive=False)

            with gr.Accordion("Fine-tuning di DeepGaze IIE su OSIE", open=True):
                gr.Markdown("Lancia `finetune_deepgaze.py` in un processo separato; con i default richiede "
                            "circa 20 minuti su CPU. MSI-Net non è addestrabile: i suoi pesi sono congelati.")
                with gr.Row():
                    ft_pattern = gr.Textbox("fixation_selection_networks", label="Unfreeze pattern")
                    ft_inspect = gr.Checkbox(False, label="Solo ispezione (elenca i parametri)")
                with gr.Row():
                    ft_max_images = gr.Number(200, precision=0, minimum=1, label="Max immagini")
                    ft_epochs = gr.Number(5, precision=0, minimum=1, label="Epoche")
                    ft_batch = gr.Number(4, precision=0, minimum=1, label="Batch size")
                    ft_lr = gr.Number(1e-5, label="Learning rate")
                with gr.Row():
                    ft_start = gr.Button("Avvia", variant="primary")
                    ft_stop = gr.Button("Interrompi", variant="stop")
                ft_log = gr.Textbox(label="Log", lines=18, max_lines=18, autoscroll=True, interactive=False)

    # tab 1
    base_inputs = [selected_path, sigma_frac, sr_sigma, sr_size, alpha, view, show_clicks]
    base_outputs = [out_original, out_center, out_itti, out_sr, info]
    deep_inputs = [selected_path, alpha, view, deep_selected]
    deep_outputs = [out_msinet, out_deepgaze, out_deepgaze_ft, deep_info]

    def on_select(evt: gr.SelectData):
        return str(IMAGE_PATHS[evt.index])

    gallery.select(on_select, None, selected_path).then(
        run, base_inputs, base_outputs).then(run_deep, deep_inputs, deep_outputs)
    upload.upload(lambda p: p, upload, selected_path).then(
        run, base_inputs, base_outputs).then(run_deep, deep_inputs, deep_outputs)
    for control in (sigma_frac, sr_sigma, sr_size, show_clicks):
        control.change(run, base_inputs, base_outputs)
    for control in (alpha, view):
        control.change(run, base_inputs, base_outputs).then(run_deep, deep_inputs, deep_outputs)
    deep_selected.change(run_deep, deep_inputs, deep_outputs)
    demo.load(run, base_inputs, base_outputs).then(run_deep, deep_inputs, deep_outputs)

    # tab 2: recomputed when the tab is opened, so it follows the image chosen in tab 1
    rank_inputs = [selected_path, rank_model, alpha]
    rank_outputs = [rank_image, rank_table, rank_info]
    for trigger in (ranking_tab.select, rank_btn.click, rank_model.change):
        trigger(run_ranking, rank_inputs, rank_outputs)

    # tab 4
    eval_btn.click(run_evaluation, eval_models, [eval_log, eval_pixel, eval_ranking, eval_rows])

    # tab 5
    osie_inputs = [osie_dir, osie_index, osie_model, alpha]
    osie_outputs = [osie_gt, osie_pred, osie_info]
    for trigger in (osie_btn.click, osie_index.submit, osie_model.change):
        trigger(run_osie_preview, osie_inputs, osie_outputs)
    ft_event = ft_start.click(
        run_finetune,
        [osie_dir, ft_max_images, ft_epochs, ft_batch, ft_lr, ft_pattern, ft_inspect], ft_log)
    ft_stop.click(stop_finetune, None, None, cancels=[ft_event])


if __name__ == "__main__":
    demo.launch()
