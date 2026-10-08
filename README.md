# NOVA — Gaze-guided Salient Object Ranking

Course project for *Industrial Applications of Computer Vision* (University of Padova, A.Y. 2025/26).
Authors: Riccardo Schiavo, Eros Patarini.

> **Project status (work in progress).** The repository contains three classical
> saliency baselines and an interactive demo to inspect them. The evaluation pipeline, the
> ground-truth collection tool, the deep saliency models and the object-ranking stage are
> **not implemented yet**; the sections below say explicitly what exists and what is planned.

## Overview

**Problem.** Given an image, predict where people look (a saliency map) and use it to rank the
objects in the scene by how much attention they attract.

**Planned experiments.**

1. **Pixel-level evaluation** — compare predicted saliency maps against human ground truth
   with standard saliency metrics. *Status: not implemented.*
2. **Object ranking** — rank the objects in an image by their predicted saliency and compare the
   ranking with the human one using rank-correlation metrics. *Status: not implemented.*

**Currently implemented.** Three hand-crafted saliency baselines (Center Prior, Itti-Koch,
Spectral Residual) and a Gradio web app (`app.py`) that runs them side by side on the images in
`img/`.

## Input / Output

| Entry point | Input | Output |
|---|---|---|
| `app.py` | No command-line arguments. Reads every `*.jpeg` file in `img/`; in the UI you can also upload a single image. | A local web UI (default `http://127.0.0.1:7860`) showing the original image and the three saliency maps as heatmap, overlay or grayscale, plus the runtime of each method. Nothing is written to disk. |

Saliency functions (`src/baselines/`):

| Function | Input | Output |
|---|---|---|
| `center_prior_saliency(image_shape, sigma_frac=0.25)` | `(height, width)` tuple | `float32` map of shape `(H, W)` in `[0, 1]` |
| `itti_koch_saliency(image, center_scales=(2, 3, 4), deltas=(3, 4), conspicuity_scale=4)` | BGR `uint8` image `(H, W, 3)` (as loaded by OpenCV) | `float32` map of shape `(H, W)` in `[0, 1]` |
| `spectral_residual_saliency(image, sigma=3.0, target_size=(64, 64))` | BGR `(H, W, 3)` or grayscale `(H, W)` image | `float32` map of shape `(H, W)` in `[0, 1]` |

## Structure

```
.
├── app.py                         # Gradio demo: runs the three baselines on img/ side by side
├── img/                           # 23 test photographs (*.jpeg) used by the demo
├── requirements.txt
└── src/
    └── baselines/
        ├── center_prior.py        # Center Prior: anisotropic Gaussian centred in the image
        ├── itti_koch.py           # Itti-Koch: intensity/colour/orientation centre-surround model
        └── spectral_residual.py   # Spectral Residual: saliency from the log-amplitude spectrum
```

## Installation & Usage

### Environment setup (step by step)

These steps were tested on Linux with Python 3.10.12, in a fresh virtual environment.

**1. Requirements.** Python ≥ 3.10 with `venv` and `pip`, plus `git`. On Debian/Ubuntu, if
`python3 -m venv` fails with *"ensurepip is not available"*, install the venv package first:

```bash
sudo apt install python3-venv
```

**2. Clone the repository.**

```bash
git clone https://github.com/riccardoschiavo/nova.git
cd nova
```

**3. Create and activate a virtual environment** (`venv/` is already in `.gitignore`).

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

The prompt now starts with `(venv)`. Every later command must be run with the environment
activated. To leave it, run `deactivate`.

**4. Install the dependencies.**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins only `numpy<2`, so pip chooses compatible versions of the other
packages. In the tested fresh install it resolved to numpy 1.26.4, opencv-python 4.11.0.86
and gradio 6.29.1.

**5. Check the installation.**

```bash
python -c "import cv2, gradio, numpy; print(numpy.__version__, cv2.__version__, gradio.__version__)"
```

**6. Make sure the test images are present.** The demo reads the `*.jpeg` files in `img/`
(23 photographs in the authors' copy).

**Troubleshooting.**

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run the commands from the repository root, not from inside `src/`. |
| `ModuleNotFoundError: No module named 'cv2'` (or `gradio`) | The virtual environment is not active: run `source venv/bin/activate`. |
| Port 7860 already in use | Start on another port: `GRADIO_SERVER_PORT=7861 python app.py`. |

### Running the demo

From the repository root, with the environment active:

```bash
python app.py
```

Then open `http://127.0.0.1:7860` in a browser. Stop the server with `Ctrl+C`.

**Environment notes.**

- Images are downscaled so that their longer side is at most 800 px before computing saliency,
  to keep the UI responsive on 12 MP photos.
- The demo was also run with numpy 2.2.6, opencv-python 5.0.0.93 and gradio 6.20.0 outside the
  pinned environment.

## Usage tutorial

**1. Pick an image.** Click a thumbnail in the gallery on the left, or drag your own image
into the upload box under it. The four panels on the right show the original image and the
maps from Center Prior, Itti-Koch and Spectral Residual. The line above them shows the image
size and the runtime of each method.

**2. Choose how to view the maps** with the *Visualizzazione* control:

- **Overlay**: the heatmap blended on the photo. *Opacità overlay* sets how much of the
  heatmap is visible. This is the easiest view for seeing *which objects* a method highlights.
- **Heatmap**: the map alone in the JET colour map. Red means most salient, blue least salient.
- **Grayscale**: the raw map in `[0, 1]` shown as 0–255. It is the actual model output, with no
  colour mapping.

**3. Explore the parameters** (the *Center prior* and *Spectral residual* panels). The maps
are recomputed whenever a control changes.

- **Center Prior, `sigma_frac`**: the width of the Gaussian relative to the image size. Small
  values give a tight central spot; large values give an almost flat map. The map does not
  depend on the image content: compare it with the other two methods to see how much of their
  output could be explained by centre bias alone.
- **Spectral Residual, `target_size`**: the resolution at which the spectrum is analysed.
  Small sizes (64, the value in the original paper) highlight whole objects; larger sizes
  respond to finer details and textures.
- **Spectral Residual, `sigma`**: the final Gaussian smoothing. Larger values give smoother,
  more blob-like maps.
- **Itti-Koch**: no controls, because it uses the fixed scales of the original paper.

**4. Things to look at.**

- **Agreement**: do the three methods agree on the most salient region?
- **Failure modes**: look for textured backgrounds (Spectral Residual), image borders
  (Itti-Koch) and centred compositions (Center Prior).
- **Speed**: compare the runtimes.

**5. Use the models from Python.** All baselines share the same output format: a `float32`
map in `[0, 1]` with the same height and width as the input.

```python
import cv2
from src.baselines.center_prior import center_prior_saliency
from src.baselines.itti_koch import itti_koch_saliency
from src.baselines.spectral_residual import spectral_residual_saliency

image = cv2.imread("img/1.jpeg")                    # BGR uint8
s_center = center_prior_saliency(image.shape[:2])
s_itti = itti_koch_saliency(image)
s_sr = spectral_residual_saliency(image)

heatmap = cv2.applyColorMap((s_itti * 255).astype("uint8"), cv2.COLORMAP_JET)
cv2.imwrite("itti_1.png", heatmap)
```

Large photos are slow with Itti-Koch at full resolution. Downscale them first, as the demo does
(longer side 800 px).

## Models

| Model | Type | Status | Reused / adapted / from scratch |
|---|---|---|---|
| Center Prior | Trivial baseline (fixed Gaussian) | Implemented | Written from scratch |
| Itti-Koch | Classical bottom-up saliency | Implemented | Written from scratch following Itti et al. (1998), without the WTA/IOR attention-shift stage; see below |
| Spectral Residual | Classical frequency-domain saliency | Implemented | Written from scratch following Hou & Zhang (2007) |
| Deep saliency models (e.g. DeepGaze, MSI-Net) | Deep learning | Planned, not in the repo | — |
| Object detector (e.g. YOLOv8) for object ranking | Deep learning | Planned, not in the repo | — |

## Ground truth & annotation procedure

*Not implemented yet.* The plan is to build an evaluation dataset from scratch by collecting
human clicks with a dedicated annotation tool. The tool, the number and order of clicks, the
Gaussian sigma used to turn clicks into maps, and the number of annotators will be documented
here once they exist. The 23 images in `img/` are currently used only by the demo; no ground
truth exists for them.

## Baselines

- **Center Prior** (`src/baselines/center_prior.py`): an anisotropic Gaussian centred in the
  image, with standard deviation `sigma_frac × width` horizontally and `sigma_frac × height`
  vertically (default `sigma_frac = 0.25`), min-max normalised to `[0, 1]`. It models the
  photographer's/viewer's centre bias and needs no image content.
- **Itti-Koch** and **Spectral Residual** are also classical baselines for the planned deep models.

None of the baselines has been evaluated quantitatively yet.

## Metrics

Planned, **not implemented yet**:

- **CC** (Pearson's linear correlation coefficient): linear correlation between the predicted map and the ground-truth density map.
- **NSS** (Normalized Scanpath Saliency): mean value of the z-scored predicted map at the ground-truth fixation/click locations.
- **AUC-Judd**: area under the ROC curve that uses the saliency values at fixated pixels as thresholds, treating fixated pixels as positives and all other pixels as negatives.
- **Spearman's ρ**: Pearson correlation between the ranks of the objects in the predicted and human rankings.
- **Kendall's τ-b**: rank correlation based on concordant vs discordant pairs of objects, with a correction for ties.

## Results

*No results yet.* The evaluation scripts have not been written, so no experiment has been run.
Tables for both experiments will be added here from the actual output files.

## Sample images

*None yet.* Example outputs (saliency overlays and heatmaps, ranked object boxes) will be saved
in the repository and referenced here. For now, use `python3 app.py` to inspect the baseline
outputs interactively.

## Datasets & external resources

- `img/`: 23 JPEG photographs (about 33 MB) used by the demo.
- Planned external resources (e.g. the OSIE dataset, pretrained model weights) will be linked
  here and **not** stored in the repository.

## Third-party code & licences

Libraries currently used:

| Component | Use | Licence |
|---|---|---|
| [NumPy](https://numpy.org/) |
| [OpenCV](https://opencv.org/) (`opencv-python`) |
| [Gradio](https://www.gradio.app/) |

No third-party model code or weights are included yet.

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
- **Demo app (`app.py`)**: original; see the AI/LLM declaration.

## References

- L. Itti, C. Koch, E. Niebur. *A Model of Saliency-Based Visual Attention for Rapid Scene
  Analysis.* IEEE TPAMI, 20(11), 1998.
- X. Hou, L. Zhang. *Saliency Detection: A Spectral Residual Approach.* CVPR, 2007.
- T. Judd, K. Ehinger, F. Durand, A. Torralba. *Learning to Predict Where Humans Look.* ICCV, 2009
  (centre bias as a saliency baseline).

## Known limitations

- Only the three classical baselines and the demo exist. The ground truth, metrics, evaluation,
  deep models and object ranking are future work.
- No quantitative evaluation has been performed.
- Itti-Koch maps are coarse, as in the paper: the final map is computed at scale 4
  (1/16 of the input resolution) and then upsampled.
- Itti-Koch maps can show spurious high values along the image borders, caused by the
  filtering and centre-surround differences at the borders.
- The demo shows the saliency maps at the downscaled resolution (longer side 800 px) and does
  not resize them back to the original image size.
- The demo only loads `*.jpeg` files from `img/`; `.jpg`/`.png` files there are ignored.
  The gallery ordering assumes numeric file names: mixing numeric and non-numeric names makes
  the app fail at start-up.
- `requirements.txt` pins only `numpy<2`. The other versions are chosen by pip at install time,
  so a future release of OpenCV or Gradio could change the behaviour.