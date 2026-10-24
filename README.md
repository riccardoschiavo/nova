# NOVA — Gaze-guided Salient Object Ranking

Course project for *Industrial Applications of Computer Vision* (University of Padova, A.Y. 2025/26).
Authors: Riccardo Schiavo, Eros Patarini.

> **Project status (work in progress).** The full pipeline is implemented: three classical
> saliency baselines, two pretrained deep models (DeepGaze IIE, MSI-Net) plus a DeepGaze IIE
> fine-tuned on OSIE, a click-based ground-truth collection tool, a pixel-level evaluation and a
> YOLOv8-based object ranking.

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

**1. Requirements.** Python ≥ 3.10 with `venv` and `pip`, plus `git`. On Debian/Ubuntu, if
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
| MSI-Net | Deep saliency (TensorFlow) | Implemented | Pretrained SALICON model from Kaggle / TF Hub; wrapper written by us |
| MSI-Net (fine-tuned) | Deep saliency (TensorFlow) | Training script only | `finetune_msinet.py` saves the trained variables, but they are not loaded by any model yet |
| YOLOv8n | Object detector for the ranking | Implemented | Pretrained COCO model from `ultralytics`, used as is |

**DeepGaze IIE.** We pass a uniform centre bias (`log(1/(H·W))` at every pixel) so that the
model does not get a free centre prior, and convert its log-density output to a density
normalised to `[0, 1]`.

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

Current data: **8 images** (`1, 10, 11, 12, 13, 14, 15, 21.jpeg`), 3 clicks each, from a single
annotator.

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
  thresholds, with clicked pixels as positives and all other pixels as negatives. 0.5 = chance,
  1 = perfect.

Object ranking (`rank_objects.py`):

- **Box score**: mean of the saliency map inside the YOLO box.
- **Spearman's ρ**: Pearson correlation between the ranks of the objects in the predicted and
  human rankings.
- **Kendall's τ-b**: rank correlation based on concordant vs discordant pairs of objects, with a
  correction for ties.

An image is used for the ranking only if YOLO finds at least 2 objects and at least 2 of them
contain a click.

## Results

Preliminary results on the 8 annotated images (CPU run, 24 Oct 2026). Mean over the images;
best value in bold.

**Experiments.**

1. **Pixel-level evaluation** (`evaluate.py`) — compare predicted saliency maps against the
   human clicks with CC, NSS and AUC-Judd.
2. **Object ranking** (`rank_objects.py`) — detect objects with YOLOv8, rank them by the mean
   predicted saliency inside each box, and compare the ranking with the order of the human
   clicks using Spearman's ρ and Kendall's τ-b.

**Experiment 1 — pixel level** (8 images):

| Model | CC | NSS | AUC-Judd |
|---|---|---|---|
| Center Prior | 0.2765 | 0.8782 | 0.8260 |
| Itti-Koch | 0.0328 | 0.1044 | 0.6418 |
| Spectral Residual | 0.1865 | 0.7969 | 0.7471 |
| DeepGaze IIE | **0.4289** | **1.9223** | **0.9036** |
| DeepGaze IIE (fine-tuned) | *to be re-run* | | |
| MSI-Net | 0.2011 | 0.5810 | 0.7689 |

**Experiment 2 — object ranking** (4 of the 8 images are usable: 10, 12, 14, 21):

| Model | Spearman | Kendall |
|---|---|---|
| Center Prior | 0.0264 | 0.0000 |
| Itti-Koch | 0.0847 | 0.0923 |
| Spectral Residual | -0.0207 | -0.0195 |
| DeepGaze IIE | **0.5264** | **0.5000** |
| DeepGaze IIE (fine-tuned) | *to be re-run* | |
| MSI-Net | 0.2372 | 0.2236 |

**Observations.**

- DeepGaze IIE is the best model in both experiments, and the only one clearly above the
  Center Prior at pixel level.
- MSI-Net is below the Center Prior at pixel level but above it in the ranking.
- Itti-Koch is close to chance at pixel level (AUC-Judd 0.64).
- The ranking numbers are very noisy: only 4 images are used, with 2–5 clicked boxes each, and
  with 2 boxes Spearman can only be +1 or −1.
- The fine-tuned DeepGaze IIE row is missing because a bug in `src/models/deepgaze.py` made it
  fail during this run; the bug is fixed and the experiments must be re-run.

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

| Component | Use | Licence |
|---|---|---|
| [NumPy](https://numpy.org/) | Arrays | BSD-3-Clause |
| [OpenCV](https://opencv.org/) (`opencv-python`) | Image I/O and processing | Apache-2.0 |
| [Gradio](https://www.gradio.app/) | Demo and annotation UIs | Apache-2.0 |
| [SciPy](https://scipy.org/) | Reading OSIE `.mat` files, Spearman/Kendall | BSD-3-Clause |
| [PyTorch](https://pytorch.org/) | DeepGaze IIE | BSD-3-Clause |
| [DeepGaze](https://github.com/matthias-k/DeepGaze) (`deepgaze_pytorch`) | DeepGaze IIE model and weights | See the upstream repository |
| [TensorFlow](https://www.tensorflow.org/) and `tensorflow_hub` | MSI-Net | Apache-2.0 |
| [MSI-Net](https://github.com/alexanderkroner/saliency) | Pretrained SALICON model | See the upstream repository |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLOv8n detector and weights | AGPL-3.0 |
| [OSIE](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels) | Fine-tuning data | See the upstream repository |

No third-party code or weights are stored in this repository.

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

- **Small ground truth**: 8 images, 3 clicks each, one annotator. The ranking uses only 4 of
  them. The results are preliminary.
- **Clicks are not fixations**: clicks are deliberate choices in order of importance, not eye
  movements, so the ground truth measures importance more than free-viewing attention.
- **Ranking depends on YOLO**: objects not in the COCO classes, or missed by YOLOv8n, cannot be
  ranked; overlapping boxes can receive the same click.
- **Fine-tuning**: small (200 images, 5 epochs, CPU), with no validation split, and the loss
  decreased only slightly. The fine-tuned MSI-Net weights are not used by any model yet.
- **No GPU**: everything runs on CPU, and the deep models take seconds per image.
- **Itti-Koch** maps are coarse (computed at 1/16 of the input resolution) and can show
  spurious high values along the image borders.
- **Demo**: the maps are shown at the downscaled resolution (longer side 800 px). Only `*.jpeg`
  files in `img/` are listed, and the gallery ordering assumes numeric file names. If
  `deepgaze_finetuned.pt` is retrained, delete `.cache/saliency/` so the demo does not show the
  old maps.
- `requirements.txt` does not pin versions, and also lists `git+https://github.com/openai/CLIP.git`,
  which no script uses at the moment.
