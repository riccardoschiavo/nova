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
            "MSI-Net needs: pip install tensorflow tensorflow_hub"
        )

# Return the model loaded in memory; load it if it has not been loaded yet.
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

# Redundant with _normalize in msi_net.py, but kept for compatibility with the rest of the code.
def _normalize(saliency_map):
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)


def msinet_saliency(image_bgr):
    _check_deps()
    model = _load_model() # Load the model.

    h, w = image_bgr.shape[:2] # Save the original image dimensions.

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB for OpenCV.

    image_resized = cv2.resize(image_rgb, (_INPUT_W, _INPUT_H)) 

    tensor = image_resized.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, ...]

    # Import TensorFlow and convert the tensor to a TensorFlow tensor.
    import tensorflow as tf
    tensor = tf.constant(tensor)
    outputs = model(tensor) # Run the model.

    if isinstance(outputs, dict):
        output = list(outputs.values())[0].numpy()
    else:
        output = outputs.numpy()

    saliency = np.squeeze(output)
    saliency = cv2.resize(saliency, (w, h))
    saliency = _normalize(saliency)
    return saliency
