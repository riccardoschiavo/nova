import numpy as np
import cv2

_INPUT_H, _INPUT_W = 240, 320
_model = None


def is_available() -> bool:
    try:
        import tensorflow          # noqa: F401
        import tensorflow_hub      # noqa: F401
        return True
    except ImportError:
        return False


def _check_deps():
    if not is_available():
        raise ImportError(
            "MSI-Net richiede: pip install tensorflow tensorflow_hub"
        )

# ritorna il modello caricato in memoria, se non è già stato caricato lo carica
def _load_model():
    global _model
    if _model is None:
        _check_deps()
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        import tensorflow_hub as hub
        _model = hub.load(
            "https://www.kaggle.com/models/alexanderkroner/msi-net/tensorFlow2/salicon/1"
        ).signatures["serving_default"]
    return _model

# ridondante con _normalize in msi_net.py, ma lo lascio per compatibilità con il resto del codice
def _normalize(saliency_map):
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)


def msinet_saliency(image_bgr):
    _check_deps()
    model = _load_model() #carico il modello 

    h, w = image_bgr.shape[:2] # salvo le dimensioni dell'immagine originale

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # converto l'immagine da BGR a RGB per OpenCV

    image_resized = cv2.resize(image_rgb, (_INPUT_W, _INPUT_H)) 

    tensor = image_resized.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, ...]

    # importo tensorflow e converto il tensore in un tensore di tensorflow
    import tensorflow as tf
    tensor = tf.constant(tensor)
    outputs = model(tensor) # eseguo 

    if isinstance(outputs, dict):
        output = list(outputs.values())[0].numpy()
    else:
        output = outputs.numpy()

    saliency = np.squeeze(output)
    saliency = cv2.resize(saliency, (w, h))
    saliency = _normalize(saliency)
    return saliency
