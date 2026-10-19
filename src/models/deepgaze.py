from xml.parsers.expat import model

import numpy as np
import cv2
from torch import device

# model is not loaded until it is needed (lazy loading)
_model = None
_current_device = None # current device

# check whether the dependencies are available
def is_available() -> bool:
    try:
        import torch              # noqa: F401
        import deepgaze_pytorch   # noqa: F401
        return True
    except ImportError:
        return False

# check and raise ImportError if dependencies are missing
def _check_deps():
 
    if not is_available():
        raise ImportError(
            "DeepGaze IIE needs:\n"            
            "  pip install git+https://github.com/matthias-k/DeepGaze.git\n"            
        )

# load the (pretrained) DeepGaze IIE model and return it
def _load_model(device):    
    global _model, _current_device
    if _model is None or _current_device != device:
        _check_deps()
        import torch
        import deepgaze_pytorch

        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)   # if this fails...
        model.eval()
        _model = model                 # ...these two lines are never executed,
        _current_device = device       # so the globals stay consistent (old + old)
        
    return _model

# build a uniform center bias so that no image region is favored
def build_uniform_centerbias(height, width):
    n = height * width
    return np.full((height, width), np.log(1.0 / n), dtype=np.float32)  # matrix of log(1/n) values

# same normalization used for evaluation
def _normalize(saliency_map):
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)

def deepgaze_saliency(image_bgr, device="cpu"):

    _check_deps()
    import torch

    # load model
    model = _load_model(device)
    h, w = image_bgr.shape[:2] # image array    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)# BGR -> RGB

    # NCHW float32 tensor
    image_tensor = torch.tensor(
        image_rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32),
        device=device,
    )

    # Uniform center bias, shape (1, h, w)
    centerbias_np = build_uniform_centerbias(h, w)
    centerbias_tensor = torch.tensor(
        centerbias_np[np.newaxis, ...],
        device=device,
    )

    # no gradient tracking; compute the log-density, then the probability density
    with torch.no_grad():
        log_density = model(image_tensor, centerbias_tensor)
    log_density = log_density.squeeze()
    density = torch.exp(log_density - log_density.max())
    
    saliency = density.cpu().numpy()
    saliency = _normalize(saliency)

    return saliency
