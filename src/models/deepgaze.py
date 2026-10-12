from xml.parsers.expat import model

import numpy as np
import cv2
from torch import device

# modello non caricato aspetto quando necessario
_model = None
_current_device = None # dispositivo corrente

# controlla se le dipendenze sono disponibili
def is_available() -> bool:
    try:
        import torch              # noqa: F401
        import deepgaze_pytorch   # noqa: F401
        return True
    except ImportError:
        return False

# controllo e solleva ImportError se le dipendenze mancano
def _check_deps():
 
    if not is_available():
        raise ImportError(
            "DeepGaze IIE needs:\n"            
            "  pip install git+https://github.com/matthias-k/DeepGaze.git\n"            
        )

# carica il modello DeepGaze IIE (pre-addestrato) e lo restituisce
def _load_model(device):    
    global _model, _current_device
    if _model is None or _current_device != device:
        _check_deps()
        import torch
        import deepgaze_pytorch

        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)   # se fallisce qui...
        model.eval()
        _model = model                 # ...queste due righe non vengono mai eseguite,
        _current_device = device       # quindi i globali restano coerenti (vecchio + vecchio)
        
    return _model

# crea un center bias uniforme per non dare vantaggio a nessuna regione dell'immagine
def build_uniform_centerbias(height, width):
    n = height * width
    return np.full((height, width), np.log(1.0 / n), dtype=np.float32)  #matrice di valori log(1/n)

# stessa normalizzazione al fine di valutazione
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

    #carica modello
    model = _load_model(device)
    h, w = image_bgr.shape[:2] #array img    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)# BGR -> RGB

    # Tensore NCHW float32
    image_tensor = torch.tensor(
        image_rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32),
        device=device,
    )

    # Center bias uniforme, shape (1, h, w)
    centerbias_np = build_uniform_centerbias(h, w)
    centerbias_tensor = torch.tensor(
        centerbias_np[np.newaxis, ...],
        device=device,
    )

    # non registra gradienti, calcola la densità di log e poi la densità di probabilità
    with torch.no_grad():
        log_density = model(image_tensor, centerbias_tensor)
    log_density = log_density.squeeze()
    density = torch.exp(log_density - log_density.max())
    
    saliency = density.cpu().numpy()
    saliency = _normalize(saliency)

    return saliency
