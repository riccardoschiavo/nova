"""Interactive demo of the saliency baselines on the images in img/.

Run with:  python app.py   (then open http://127.0.0.1:7860)
"""
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from src.baselines.center_prior import center_prior_saliency
from src.baselines.itti_koch import itti_koch_saliency
from src.baselines.spectral_residual import spectral_residual_saliency

IMG_DIR = Path(__file__).parent / "img"
IMAGE_PATHS = sorted(IMG_DIR.glob("*.jpeg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

# Downscale large images before processing so the UI stays responsive
MAX_SIDE = 800


def load_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def to_heatmap(saliency):
    # saliency in [0, 1] -> RGB heatmap
    colored = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def to_overlay(image_bgr, saliency, alpha):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, to_heatmap(saliency), alpha, 0)


def run(path, sigma_frac, sr_sigma, sr_size, alpha, view):
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

    render = {
        "Heatmap": lambda m: to_heatmap(m),
        "Overlay": lambda m: to_overlay(image, m, alpha),
        "Grayscale": lambda m: (m * 255).astype(np.uint8),
    }[view]

    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    info = (f"**{Path(path).name}** — {image.shape[1]}×{image.shape[0]} px  \n"
            + " · ".join(f"{name}: {sec * 1000:.0f} ms" for name, sec in timings.items()))
    return (original,
            render(maps["Center prior"]),
            render(maps["Itti-Koch"]),
            render(maps["Spectral residual"]),
            info)


with gr.Blocks(title="Nova – Saliency baselines") as demo:
    gr.Markdown("# Saliency baselines\nSeleziona un'immagine dalla galleria per confrontare i tre metodi.")
    selected_path = gr.State(str(IMAGE_PATHS[0]) if IMAGE_PATHS else None)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gallery = gr.Gallery(
                value=[str(p) for p in IMAGE_PATHS], label="img/", columns=3,
                height=420, allow_preview=False, object_fit="cover",
            )
            upload = gr.Image(label="…oppure carica un'immagine", type="filepath", height=160)

            view = gr.Radio(["Heatmap", "Overlay", "Grayscale"], value="Overlay", label="Visualizzazione")
            alpha = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Opacità overlay")

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

    params = [sigma_frac, sr_sigma, sr_size, alpha, view]
    outputs = [out_original, out_center, out_itti, out_sr, info]

    def on_select(evt: gr.SelectData):
        return str(IMAGE_PATHS[evt.index])

    gallery.select(on_select, None, selected_path).then(run, [selected_path, *params], outputs)
    upload.upload(lambda p: p, upload, selected_path).then(run, [selected_path, *params], outputs)
    for control in params:
        control.change(run, [selected_path, *params], outputs)
    demo.load(run, [selected_path, *params], outputs)


if __name__ == "__main__":
    demo.launch()
