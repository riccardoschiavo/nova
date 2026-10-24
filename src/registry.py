from src.baselines.center_prior import center_prior_saliency
from src.baselines.spectral_residual import spectral_residual_saliency
from src.baselines.itti_koch import itti_koch_saliency
from src.models.msi_net import msinet_saliency
from src.models.deepgaze import deepgaze_saliency, deepgaze_finetuned_saliency


# name -> (function, needs_full_image)
# NOTE: PyTorch models (DeepGaze) MUST come before TensorFlow ones (MSI-Net):
# on systems without a GPU, TF's CUDA initialization corrupts the runtime and
# causes a segfault when PyTorch tries to initialize afterwards.
_REGISTRY = {
    "Center Prior": (center_prior_saliency, False),
    "Spectral Residual": (spectral_residual_saliency, True),
    "Itti-Koch": (itti_koch_saliency, True),
    "DeepGaze IIE": (deepgaze_saliency, True),
    "DeepGaze IIE (fine-tuned)": (deepgaze_finetuned_saliency, True),
    "MSI-Net": (msinet_saliency, True),
}

AVAILABLE_MODELS = list(_REGISTRY.keys())


def compute_saliency(model_name, image_bgr):
    if model_name not in _REGISTRY:
        raise KeyError(
            f"Unknown model: '{model_name}'. "
            f"Available: {', '.join(AVAILABLE_MODELS)}"
        )

    model_fn, needs_image = _REGISTRY[model_name]
    if needs_image:
        return model_fn(image_bgr)
    return model_fn(image_bgr.shape[:2])
