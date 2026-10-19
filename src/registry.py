from src.baselines.center_prior import center_prior_saliency
from src.baselines.spectral_residual import spectral_residual_saliency
from src.baselines.itti_koch import itti_koch_saliency
from src.models.msi_net import msinet_saliency
from src.models.deepgaze import deepgaze_saliency, deepgaze_finetuned_saliency

# list of the models
_REGISTRY = {
    "Center Prior": (center_prior_saliency, False),
    "Spectral Residual": (spectral_residual_saliency, True),
    "Itti-Koch": (itti_koch_saliency, True),
    "DeepGaze IIE": (deepgaze_saliency, True),
    "MSI-Net": (msinet_saliency, True),
}

AVAILABLE_MODELS = list(_REGISTRY.keys()) # available models in order


def compute_saliency(model_name, image_bgr):
    #error message if the model is not available
    if model_name not in _REGISTRY:
        raise KeyError(
            f"Unknown model: '{model_name}'. "
            f"Available: {', '.join(AVAILABLE_MODELS)}"
        )
    # model function, image needed or not
    model_fn, needs_image = _REGISTRY[model_name]
    if needs_image:
        return model_fn(image_bgr) # compute saliency with the full image
    return model_fn(image_bgr.shape[:2]) # compute saliency with only the image size for center prior
