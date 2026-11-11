# NOVA — Gaze-guided Salient Object Ranking

Course project for *Industrial Applications of Computer Vision* (University of Padova, A.Y. 2025/26).
Authors: Riccardo Schiavo, Eros Patarini.

> **Project status.** The full pipeline is implemented: three classical saliency baselines, two
> pretrained deep models (DeepGaze IIE, MSI-Net) plus a DeepGaze IIE fine-tuned on OSIE, a
> click-based ground-truth collection tool, a pixel-level evaluation and a YOLOv8-based object
> ranking. Final results on 23 annotated images are in [Results](#results).

## Overview

**Problem.** Given an image, predict where people look (a saliency map) and use it to rank the
objects in the scene by how much attention they attract.

## Pipeline

```
image ──► saliency model ──► saliency map ──┬──► evaluate.py      (CC, NSS, AUC-Judd vs clicks)
                                            │
image ──► YOLOv8n ──► object boxes ─────────┴──► rank_objects.py  (box score = mean saliency,
                                                                   Spearman / Kendall vs click order)
clicks.json ◄── collect.py (human annotator)
```

## Input / Output

| Entry point | Input | Output |
|---|---|---|
| `app.py` | Every `*.jpeg` file in `img/`; in the UI you can also upload a single image. | Local web UI (`http://127.0.0.1:7860`) with the original image, the 3 baselines and the 3 deep models as heatmap, overlay or grayscale, plus the runtime of each method. Deep maps are cached in `.cache/saliency/`. |
| `collect.py` | `--images` (default `img/`), `--out` (default `clicks.json`). | Web UI to click the 3–6 most important points of each image, in order of importance. Clicks are saved to `clicks.json`. |
| `evaluate.py` | `--images` (default `img/`), `--clicks` (default `clicks.json`). | Table with mean CC, NSS and AUC-Judd per model, printed to the terminal. |
| `rank_objects.py` | `--images` (default `img/`), `--clicks` (default `clicks.json`). | Table with mean Spearman and Kendall per model and the number of images used, printed to the terminal. |
| `finetune_deepgaze.py` | `--osie <OSIE repo folder>`, `--unfreeze-pattern`, training options. | `deepgaze_finetuned.pt` (full state dict of DeepGaze IIE). |
| `finetune_msinet.py` | `--osie <OSIE repo folder>`, `--unfreeze-pattern`, training options. | `msinet_finetuned/finetuned_weights.npz` (trained variables only). |

All saliency models share the same interface through `src/registry.py`:
`compute_saliency(model_name, image_bgr)` returns a `float32` map of shape `(H, W)` in `[0, 1]`.
`AVAILABLE_MODELS` lists the registered names.

## Structure

```
.
├── app.py                    # Gradio demo: all models side by side on img/
├── collect.py                # Gradio tool to collect the ground-truth clicks
├── clicks.json               # collected clicks: {image: [{x, y, order}, ...]}
├── evaluate.py               # experiment 1: pixel-level metrics vs clicks
├── rank_objects.py           # experiment 2: YOLOv8 object ranking vs click order
├── finetune_deepgaze.py      # fine-tuning of DeepGaze IIE on OSIE
├── finetune_msinet.py        # fine-tuning of MSI-Net on OSIE
├── osie_ground_truth.py      # reads OSIE fixations.mat and builds Gaussian heatmaps
├── img/                      # 23 test photographs (*.jpeg)
├── requirements.txt
└── src/
    ├── registry.py           # model name -> saliency function
    ├── baselines/
    │   ├── center_prior.py       # Center Prior: anisotropic Gaussian centred in the image
    │   ├── itti_koch.py          # Itti-Koch: intensity/colour/orientation centre-surround model
    │   └── spectral_residual.py  # Spectral Residual: saliency from the log-amplitude spectrum
    └── models/
        ├── deepgaze.py       # DeepGaze IIE (pretrained and fine-tuned), PyTorch
        └── msi_net.py        # MSI-Net (SALICON weights from Kaggle/TF Hub), TensorFlow
```

Not stored in the repository (see `.gitignore`): `venv/`, `.cache/`, model weights (`*.pt`:
`deepgaze_finetuned.pt`, `yolov8n.pt`) and the OSIE clone `predicting-human-gaze-beyond-pixels/`.

## Installation & Usage

### Environment setup

Tested on Linux (Ubuntu) with Python 3.13, CPU only (no GPU).

**1. Requirements.** Python ≥ 3.11 with `venv` and `pip`, plus `git`. On Debian/Ubuntu, if
`python3 -m venv` fails with *"ensurepip is not available"*, run `sudo apt install python3-venv`.

**2. Clone the repository and create a virtual environment.**

```bash
git clone https://github.com/riccardoschiavo/nova.git
cd nova
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Every later command must be run from the repository root with the environment active.

**3. Install the dependencies.**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` does not pin versions (except `setuptools<81`, needed by `tensorflow_hub`).
Tested versions: numpy 2.5.4, opencv-python 4.11.0, gradio 6.30.0, scipy 1.18.1,
torch 2.14.1 (CPU), tensorflow 2.21.0, ultralytics 8.4.185.

**4. Weights downloaded automatically on first use.**

- DeepGaze IIE and its backbones: into `~/.cache/torch/`.
- MSI-Net: from Kaggle / TF Hub.
- YOLOv8n: `yolov8n.pt` (about 6 MB) into the repository root.

**5. (Only for fine-tuning) Get the OSIE dataset.**

```bash
git clone https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels.git
```

The scripts read `data/eye/fixations.mat` and `data/stimuli/` from that folder.

### Running the scripts

```bash
python app.py                  # demo, then open http://127.0.0.1:7860
python collect.py              # collect clicks into clicks.json
python evaluate.py             # experiment 1: pixel-level metrics
python rank_objects.py         # experiment 2: object ranking
```

Run `evaluate.py` and `rank_objects.py` one after the other, not in parallel: both load all the
models, and running them together can exhaust the RAM.

**Fine-tuning DeepGaze IIE** (about 20 minutes on CPU with the defaults):

```bash
# list the parameter names to choose what to train
python finetune_deepgaze.py --osie predicting-human-gaze-beyond-pixels --inspect-only

# train only the last readout block of each ensemble member
python finetune_deepgaze.py --osie predicting-human-gaze-beyond-pixels \
    --unfreeze-pattern fixation_selection_networks \
    --max-images 200 --epochs 5 --batch-size 4 --learning-rate 1e-5
```

The weights are saved to `deepgaze_finetuned.pt` and loaded by the
`DeepGaze IIE (fine-tuned)` model.

**Troubleshooting.**

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run the commands from the repository root. |
| `ModuleNotFoundError` for `cv2`, `gradio`, `torch`, … | The virtual environment is not active: `source venv/bin/activate`. |
| `failed call to cuInit` / `Could not find cuda drivers` | Harmless: TensorFlow looks for a GPU, the scripts force the CPU on purpose. Use `TF_CPP_MIN_LOG_LEVEL=3 python …` to hide these logs. |
| `Fine-tuned weights not found` | Run `finetune_deepgaze.py` first; the other models keep working. |
| Segfault when both TensorFlow and PyTorch models run | PyTorch models must run before TensorFlow ones. `src/registry.py` keeps this order; `app.py` runs each deep model in a separate process. |
| Port 7860 already in use | `GRADIO_SERVER_PORT=7861 python app.py`. |

## Usage tutorial

**1. Pick an image.** Click a thumbnail in the gallery on the left, or drag your own image into
the upload box. The panels on the right show the original image, the three baselines and,
below, the three deep models. Use the checkboxes to enable or disable each deep model: the
first run downloads the weights and can take tens of seconds; later runs use the disk cache.

**2. Choose how to view the maps** with *Visualizzazione*:

- **Overlay**: the heatmap blended on the photo; *Opacità overlay* sets its opacity.
- **Heatmap**: the map alone in the JET colour map (red = most salient, blue = least).
- **Grayscale**: the raw map in `[0, 1]` shown as 0–255.

**3. Explore the baseline parameters.**

- **Center Prior, `sigma_frac`**: width of the Gaussian relative to the image size.
- **Spectral Residual, `target_size`**: resolution at which the spectrum is analysed (64 in the
  paper); **`sigma`**: final Gaussian smoothing.
- **Itti-Koch** and the deep models have no controls.

**4. Use the models from Python.**

```python
import cv2
from src.registry import AVAILABLE_MODELS, compute_saliency

image = cv2.imread("img/1.jpeg")                     # BGR uint8
saliency = compute_saliency("DeepGaze IIE", image)    # float32 (H, W) in [0, 1]

heatmap = cv2.applyColorMap((saliency * 255).astype("uint8"), cv2.COLORMAP_JET)
cv2.imwrite("deepgaze_1.png", heatmap)
```

Downscale large photos first (the scripts use a longer side of 800 px): Itti-Koch and the deep
models are slow at 12 MP.

## Models

| Model | Type | Status | Reused / adapted / from scratch |
|---|---|---|---|
| Center Prior | Trivial baseline (fixed Gaussian) | Implemented | Written from scratch |
| Itti-Koch | Classical bottom-up saliency | Implemented | Written from scratch following Itti et al. (1998), without the WTA/IOR stage |
| Spectral Residual | Classical frequency-domain saliency | Implemented | Written from scratch following Hou & Zhang (2007) |
| DeepGaze IIE | Deep saliency (PyTorch) | Implemented | Pretrained model from `deepgaze_pytorch`, used with a uniform centre bias; wrapper written by us |
| DeepGaze IIE (fine-tuned) | Deep saliency (PyTorch) | Implemented | DeepGaze IIE with the readout block fine-tuned on OSIE by us |
| MSI-Net | Deep saliency (TensorFlow) | Implemented | Pretrained SALICON model from Kaggle / TF Hub; wrapper written by us, pre-processing as in the official code (see below) |
| MSI-Net (fine-tuned) | Deep saliency (TensorFlow) | Training script only | `finetune_msinet.py` saves the trained variables, but they are not loaded by any model yet |
| YOLOv8n | Object detector for the ranking | Implemented | Pretrained COCO model from `ultralytics`, used as is |

**DeepGaze IIE.** We pass a uniform centre bias (`log(1/(H·W))` at every pixel) so that the
model does not get a free centre prior, and convert its log-density output to a density
normalised to `[0, 1]`. The centre bias enters DeepGaze IIE only in its finalizer, as an
additive term on the log-density that the final `logsumexp` normalisation removes when it is
constant: with a uniform centre bias the explicit centre prior of the model has no effect
(maps with `log(1)` and `log(0.3)` differ by at most 1e-6).

**MSI-Net.** The image is converted to RGB, resized keeping the aspect ratio to fit 320×240,
padded with the value 126 and fed with pixel values in `[0, 255]`: the network subtracts the
ImageNet mean itself (official code: [alexanderkroner/saliency](https://github.com/alexanderkroner/saliency),
`model.py` and `data.py`). The padding is cropped from the output before resizing it back.
Earlier versions of `src/models/msi_net.py` (before commit `2f83c00`) divided the pixels by 255
and stretched the image to 320×240: the network then received an almost constant input and
returned nearly the same map for every image (CC 0.998 between the maps of different images).

**Fine-tuning DeepGaze IIE on OSIE.** DeepGaze IIE is an ensemble of 4 networks (different
backbones). In each one we freeze the backbone, the saliency network and the finalizer, and
train only `fixation_selection_networks` (≈311k parameters in total). The training set is the
first 200 OSIE images, resized to 320×240. The target is the Gaussian heatmap of all the
observers' fixations (σ = 0.05 × image width). The loss is the MSE between that heatmap and
the predicted density, min-max normalised. Adam, lr 1e-5, batch 4, 5 epochs. Frozen BatchNorm
layers are kept in eval mode. The mean training loss went from 0.02774 (epoch 1) to 0.02679
(epoch 5). No validation split is used, and OSIE is not used for evaluation (the evaluation
uses our own clicks on `img/`).

## Ground truth & annotation procedure

The ground truth is collected with `collect.py`:

- the annotator sees one image at a time and clicks **3 to 6 points** on the most important
  objects or regions, **from the most to the least important**;
- clicks can be undone, and an image can be redone later;
- each click is stored in `clicks.json` with its pixel coordinates in the original image and
  its order (`1` = most important).

The clicks are then used in two ways:

- **Pixel-level evaluation**: a binary map with a 1 at each click (for NSS and AUC-Judd), and a
  heatmap obtained by summing a Gaussian with σ = 0.05 × image width at each click (for CC),
  the same construction used for OSIE.
- **Object ranking**: each YOLO box gets the order of the first click that falls inside it.
  Boxes that are never clicked are excluded.

Current data: **23 images** (all of `img/`), **92 clicks** (3–6 per image), from a **single
annotator**. The clicks are significantly closer to the image centre than uniformly placed
points (mean normalised distance 0.405 vs 0.541), and the first click is the most central.

## Baselines

- **Center Prior** (`src/baselines/center_prior.py`): an anisotropic Gaussian centred in the
  image, with standard deviation `sigma_frac × width` horizontally and `sigma_frac × height`
  vertically (default `sigma_frac = 0.25`), min-max normalised to `[0, 1]`. It models the
  centre bias and needs no image content.
- **Itti-Koch** and **Spectral Residual**: classical bottom-up baselines for the deep models
  (details below).

## Metrics

Pixel level (`evaluate.py`; maps are computed on the image downscaled to a longer side of
800 px, then resized back to the original resolution of the clicks):

- **CC** (Pearson's linear correlation coefficient): linear correlation between the predicted
  map and the click heatmap. 1 = same trend, 0 = no relation.
- **NSS** (Normalized Scanpath Saliency): mean of the z-scored predicted map at the clicked
  pixels. 0 = chance; higher is better.
- **AUC-Judd**: area under the ROC curve that uses the saliency at the clicked pixels as
  thresholds, with clicked pixels as positives and all other pixels as negatives. 1 = perfect.
  With only 3–6 clicks per image the chance level is **about 0.60, not 0.5**: the trapezoidal
  rule over the click thresholds (MIT benchmark definition) overestimates the area by
  `fp_n / (2·n_clicks)`, about `1/(2(n+1))` on a random map. A constant map gives exactly 0.5
  only through an explicit special case in the code.

Object ranking (`rank_objects.py`):

- **Box score**: mean of the saliency map inside the YOLO box.
- **Spearman's ρ**: Pearson correlation between the ranks of the objects in the predicted and
  human rankings.
- **Kendall's τ-b**: rank correlation based on concordant vs discordant pairs of objects, with a
  correction for ties.

An image is used for the ranking only if YOLO finds at least 2 objects and at least 2 of them
contain a click.

## Results

Final run on the **23 annotated images** of `img/` (CPU, 25 Oct 2026), with the protocol of
`evaluate.py` and `rank_objects.py`: saliency computed with a longer side of 800 px and resized
back to the original resolution. Mean over the images, with the 95% bootstrap confidence
interval over the images (10 000 resamples) in brackets.

**Experiment 1: pixel level** (23 images).

| Model | CC | NSS | AUC-Judd | AUC-Judd − chance |
|---|---|---|---|---|
| Center Prior | 0.273 [0.211, 0.334] | 0.676 [0.518, 0.822] | 0.770 [0.736, 0.800] | 0.167 |
| Spectral Residual | 0.297 [0.241, 0.356] | 1.033 [0.820, 1.260] | 0.812 [0.776, 0.847] | 0.210 |
| Itti-Koch | 0.142 [0.086, 0.198] | 0.702 [0.400, 1.048] | 0.730 [0.674, 0.784] | 0.127 |
| DeepGaze IIE | 0.439 [0.374, 0.498] | **1.944** [1.650, 2.241] | **0.899** [0.879, 0.919] | **0.297** |
| DeepGaze IIE (fine-tuned) | 0.437 [0.372, 0.497] | **1.959** [1.657, 2.264] | **0.900** [0.879, 0.919] | **0.297** |
| MSI-Net | **0.449** [0.385, 0.508] | 1.640 [1.398, 1.873] | 0.887 [0.861, 0.909] | 0.284 |
| *Uniform map* | 0.000 | 0.000 | 0.500 | −0.103 |
| *Random noise* | 0.000 | 0.016 | 0.606 | 0.003 |
| *Shuffled clicks of the other images* | 0.249 | 0.705 | 0.777 | 0.174 |

"AUC-Judd − chance" subtracts the chance level of AUC-Judd measured on each image with 100
random maps and the same clicks (mean 0.603, see [Metrics](#metrics)).

**Experiment 2: object ranking** (13 images; 10 are excluded: 2 with fewer than 2 YOLO boxes,
6 with fewer than 2 clicked boxes, and 7.jpeg and 16.jpeg, where a single click falls inside all
the clicked overlapping boxes and gives them the same rank). Top-1 = the box with the highest
score is the one of the first click, among the clicked boxes.

| Model / rule | Spearman | Kendall | Top-1 |
|---|---|---|---|
| Center Prior | 0.496 | 0.459 | 10/13 |
| Spectral Residual | 0.006 | −0.014 | 7/13 |
| Itti-Koch | −0.042 | −0.049 | 7/13 |
| DeepGaze IIE | **0.688** | **0.664** | **12/13** |
| DeepGaze IIE (fine-tuned) | **0.688** | **0.664** | **12/13** |
| MSI-Net | 0.380 | 0.356 | 10/13 |
| *Rule: larger box first* | 0.196 | 0.213 | 8/13 |
| *Rule: higher YOLO confidence first* | −0.004 | 0.003 | 7/13 |
| *Rule: box closer to the centre first* | 0.429 | 0.396 | 9/13 |
| *Random ranking (expected)* | 0 | 0 | 0.506 |

`rank_objects.py` reports 15 images: in 7.jpeg and 16.jpeg Spearman and Kendall are undefined
(constant ground truth) and are skipped in the mean, so the means above are the same.

**Observations** (statistics computed in the presentation, n = 23 and n = 13):

- DeepGaze IIE is above the Center Prior in all three pixel metrics (Wilcoxon signed-rank,
  Holm-corrected p ≤ 0.004, better on 19–23 of 23 images). MSI-Net and DeepGaze IIE are not
  separated (Friedman + Nemenyi, critical difference 2.19 ranks).
- The fine-tuning changes almost nothing: the fine-tuned and base maps have CC 0.9998, and no
  metric differs significantly (Wilcoxon, Holm).
- The classical models are close to the Center Prior and to the "shuffled clicks" baseline,
  which exploit only the centre bias of the clicks.
- In the ranking only DeepGaze IIE is above a random ranking (permutation test, Holm-corrected
  p = 0.013 for Spearman, 0.016 for Top-1). A Friedman test over the 6 models and the 3 rules is
  not significant (p = 0.17 for Spearman). The Center Prior ranks the boxes almost like the
  "closer to the centre" rule (Kendall τ-b 0.85 between the two).
- CC depends on the σ of the click heatmap (the order of the models changes between σ = 2.5% and
  10% of the width); NSS and AUC-Judd do not depend on σ.

## Sample images

*None yet.* Use `python app.py` to inspect the maps interactively.

## Datasets & external resources

- `img/`: 23 JPEG photographs (about 33 MB), used by the demo and, with `clicks.json`, as the
  evaluation set.
- [OSIE](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels) (Xu et al., 2014):
  700 images with eye-tracking fixations, used only for fine-tuning. Not stored in the
  repository.
- Pretrained weights (DeepGaze IIE, MSI-Net, YOLOv8n) are downloaded automatically and not
  stored in the repository. `deepgaze_finetuned.pt` (≈420 MB) is produced locally by
  `finetune_deepgaze.py`.

## Third-party code & licences

| Component | Version tested | Use | Source | Licence |
|---|---|---|---|---|
| NumPy | 2.5.4 | Arrays | [numpy.org](https://numpy.org/) | BSD-3-Clause |
| OpenCV (`opencv-python`) | 4.11.0 | Image I/O and processing | [opencv.org](https://opencv.org/) | Apache-2.0 |
| Gradio | 6.30.0 | Demo and annotation UIs | [gradio.app](https://www.gradio.app/) | Apache-2.0 |
| pandas | 3.0.6 | Tables in `app.py` (installed as a dependency of Gradio) | [pandas.pydata.org](https://pandas.pydata.org/) | BSD-3-Clause |
| SciPy | 1.18.1 | Reading OSIE `.mat` files, Spearman/Kendall | [scipy.org](https://scipy.org/) | BSD-3-Clause |
| PyTorch | 2.14.1 (CPU) | DeepGaze IIE | [pytorch.org](https://pytorch.org/) | BSD-3-Clause |
| einops | 0.8.2 | Required by `deepgaze_pytorch` | [arogozhnikov/einops](https://github.com/arogozhnikov/einops) | MIT |
| DeepGaze (`deepgaze_pytorch`) | 1.2.1, commit `c7db17e` | DeepGaze IIE model and pretrained weights | [matthias-k/DeepGaze](https://github.com/matthias-k/DeepGaze) | **No licence declared** in the repository (no LICENSE file; `license='MIT'` is commented out in `setup.py`). Used for academic coursework only, citing Linardos et al. (2021). |
| TensorFlow | 2.21.0 | MSI-Net | [tensorflow.org](https://www.tensorflow.org/) | Apache-2.0 |
| `tensorflow_hub` | 0.16.1 | Loading MSI-Net | [tensorflow/hub](https://github.com/tensorflow/hub) | Apache-2.0 |
| MSI-Net | Kaggle model `alexanderkroner/msi-net/tensorFlow2/salicon/1` | Pretrained SALICON model | [alexanderkroner/saliency](https://github.com/alexanderkroner/saliency) | MIT (code repository) |
| Ultralytics | 8.4.185 | YOLOv8n detector and COCO weights (`yolov8n.pt`) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| OSIE dataset | — | Fine-tuning data (not stored here) | [NUS-VIP/predicting-human-gaze-beyond-pixels](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels) | MIT |
| CLIP | — | Listed in `requirements.txt`, not used by any script | [openai/CLIP](https://github.com/openai/CLIP) | MIT |

No third-party code or weights are stored in this repository: the weights are downloaded at
first use (DeepGaze IIE and its backbones into `~/.cache/torch/`, MSI-Net from Kaggle / TF Hub,
`yolov8n.pt` into the repository root).

## Use of AI / LLM tools

- **Claude (Anthropic)**, used in four ways:
  - **in Antigravity**, as a coding assistant during development: it wrote parts of the project
    code and helped with debugging and explanations;
  - **in chat**, to plan the evaluation protocol and the statistical analysis, to write the
    instructions given to Claude Code, and to prepare the study notes for the oral presentation;
  - **for literature search**, to find and summarise information from the computer vision
    literature (saliency models, evaluation metrics, statistical tests), based on papers and
    other reliable sources; the papers actually used are listed in [References](#references);
  - **in Claude Code**, in the final evaluation phase (below).
- **Claude Code (Anthropic)**, in the final evaluation phase. It:
  - found and fixed the MSI-Net pre-processing bug in `src/models/msi_net.py` (commit `2f83c00`,
    see [Models](#models));
  - wrote the analysis scripts (not included in this repository) that produced the statistics,
    figures and tables of the final presentation: bootstrap confidence intervals, Friedman /
    Nemenyi and Wilcoxon tests, chance level of AUC-Judd, ranking baselines and permutation tests,
    sensitivity to σ and resolution, checks on the fine-tuned weights;
  - updated this README (results, MSI-Net, licences, limitations).

## Reused / adapted / original

- **Center Prior**: original implementation of the standard centre-bias Gaussian.
- **Itti-Koch**: original implementation of the saliency-map stage of Itti et al. (1998).
  The code follows the paper:
  - 9-level dyadic Gaussian pyramids;
  - intensity `I = (r+g+b)/3`;
  - r, g, b normalised by `I` only where `I` exceeds 1/10 of its maximum, and set to 0 elsewhere;
  - broadly tuned channels `R = r−(g+b)/2`, `G = g−(r+b)/2`, `B = b−(r+g)/2` and
    `Y = (r+g)/2 − |r−g|/2 − b`, with negative values set to 0;
  - centre-surround differences with `c ∈ {2, 3, 4}`, `s = c + δ`, `δ ∈ {3, 4}`, giving
    42 feature maps (6 intensity, 12 double-opponent colour `RG`/`BY`, 24 orientation);
  - normalisation operator N(·): rescale to `[0, M]`, then multiply by `(M − m̄)²`, where
    `m̄` is the mean of the other local maxima;
  - conspicuity maps built by across-scale addition at scale 4, with orientation normalised
    per angle before summing;
  - final map `S = (N(Ī) + N(C̄) + N(Ō)) / 3`.

  Implementation choices where the paper is not specific, or differs:
  - **Gabor pyramids**: the oriented Gabor pyramids are approximated by filtering every level
    of the intensity Gaussian pyramid with a zero-mean 15×15 Gabor kernel (σ=4, λ=10, γ=0.5)
    at 0°, 45°, 90° and 135°.
  - **Local maxima for N(·)**: found in a 7×7 neighbourhood. Flat regions are excluded, and so
    is the global maximum.
  - **Downscaling**: maps are reduced to scale 4 with area interpolation, and surround maps are
    enlarged to the centre scale with bilinear interpolation.
  - **Output**: `S` (at scale 4) is resized to the input size and min-max normalised to
    `[0, 1]`, to match the common output format of all models.
  - **Not implemented**: the winner-take-all network and inhibition of return, which simulate
    the sequence of attention shifts. They do not change the saliency map itself.
- **Spectral Residual**: original implementation of Hou & Zhang (2007): grayscale image resized
  to 64×64, FFT, log-amplitude minus its 3×3 local mean, inverse FFT with the original phase,
  squared magnitude, Gaussian blur, normalisation and resize to the input size.
- **DeepGaze IIE / MSI-Net wrappers** (`src/models/`): original code around the pretrained
  models (pre-processing, centre bias, normalisation, lazy loading).
- **Fine-tuning scripts, `osie_ground_truth.py`**: original; they use the pretrained models and
  the OSIE fixations.
- **`collect.py`, `evaluate.py`, `rank_objects.py`**: original. The metrics are implemented
  from their standard definitions; Spearman and Kendall come from SciPy; object detection uses
  Ultralytics YOLOv8n.
- **Demo app (`app.py`)**: original.

## References

- L. Itti, C. Koch, E. Niebur. *A Model of Saliency-Based Visual Attention for Rapid Scene
  Analysis.* IEEE TPAMI, 20(11), 1998.
- X. Hou, L. Zhang. *Saliency Detection: A Spectral Residual Approach.* CVPR, 2007.
- T. Judd, K. Ehinger, F. Durand, A. Torralba. *Learning to Predict Where Humans Look.* ICCV, 2009
  (centre bias as a saliency baseline, AUC-Judd).
- A. Linardos, M. Kümmerer, O. Press, M. Bethge. *DeepGaze IIE: Calibrated Prediction in and
  Out-of-Domain for State-of-the-Art Saliency Modeling.* ICCV, 2021.
- A. Kroner, M. Senden, K. Driessens, R. Goebel. *Contextual Encoder–Decoder Network for Visual
  Saliency Prediction.* Neural Networks, 129, 2020 (MSI-Net).
- J. Xu, M. Jiang, S. Wang, M. S. Kankanhalli, Q. Zhao. *Predicting Human Gaze Beyond Pixels.*
  Journal of Vision, 14(1), 2014 (OSIE).
- Z. Bylinskii, T. Judd, A. Oliva, A. Torralba, F. Durand. *What Do Different Evaluation Metrics
  Tell Us About Saliency Models?* IEEE TPAMI, 41(3), 2019.

## Known limitations

- **Small ground truth**: 23 images, 3–6 clicks each (92 in total), one annotator, so no
  inter-observer upper bound can be estimated. The ranking uses only 13 images, 7 of which have
  only 2 clicked boxes (Spearman is then +1 or −1).
- **Clicks are not fixations**: clicks are deliberate choices in order of importance, not eye
  movements, so the ground truth measures importance more than free-viewing attention.
- **Ranking depends on YOLO**: 55 of the 92 clicks (60%) fall outside every YOLO box, and in 10
  of 23 images the first click falls outside every box. Objects not in the COCO classes, or
  missed by YOLOv8n, cannot be ranked; overlapping boxes can receive the same click.
- **Box score**: the mean saliency inside the box. Other choices (max, sum, 90th percentile)
  change the results; the plain sum favours large boxes.
- **AUC-Judd**: its chance level is about 0.60 with this number of clicks (see [Metrics](#metrics)).
- **Fine-tuning**: small (200 images, 5 epochs, lr 1e-5, CPU), MSE loss instead of the
  log-likelihood used to train DeepGaze, no validation split; only 0.30% of the parameters are
  trained and the maps barely change (mean absolute difference ≈ 0.003). The fine-tuned MSI-Net
  weights are not used by any model yet.
- **Centre bias**: DeepGaze IIE is used with a uniform centre bias, which disables its explicit
  centre prior (see [Models](#models)).
- **No GPU**: everything runs on CPU, and the deep models take seconds per image
  (DeepGaze IIE ≈ 2.2 s per 800-px image).
- **Itti-Koch** maps are coarse (computed at 1/16 of the input resolution), can show spurious high
  values along the image borders, and depend on the input resolution (NSS +0.09 on average from
  800 to 1200 px).
- **Demo**: the maps are shown at the downscaled resolution (longer side 800 px). Only `*.jpeg`
  files in `img/` are listed, and the gallery ordering assumes numeric file names. Deep maps are
  cached in `.cache/saliency/` by image content only: delete that folder after changing a model
  or retraining `deepgaze_finetuned.pt`, or the demo shows the old maps.
- `requirements.txt` does not pin versions, does not list `pandas` (installed with Gradio), and
  lists `git+https://github.com/openai/CLIP.git`, which no script uses.
