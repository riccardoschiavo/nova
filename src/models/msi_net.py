import numpy as np
import cv2

_INPUT_H, _INPUT_W = 240, 320
_PAD_VALUE = 126.0 # padding value of the official data.py (github.com/alexanderkroner/saliency)
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

    # resize keeping the aspect ratio, then symmetric padding up to 240x320 (as the official data.py)
    ratio = min(_INPUT_H / h, _INPUT_W / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    image_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interpolation)
    top, left = (_INPUT_H - new_h) // 2, (_INPUT_W - new_w) // 2
    canvas = np.full((_INPUT_H, _INPUT_W, 3), _PAD_VALUE, dtype=np.float32)
    canvas[top:top + new_h, left:left + new_w] = image_resized

    # pixel values stay in [0, 255]: the graph subtracts the ImageNet mean itself,
    # dividing by 255 would give the network an almost constant image
    tensor = canvas[np.newaxis, ...]

    # Import TensorFlow and convert the tensor to a TensorFlow tensor.
    import tensorflow as tf
    tensor = tf.constant(tensor)
    outputs = model(tensor) # Run the model.

    if isinstance(outputs, dict):
        output = list(outputs.values())[0].numpy()
    else:
        output = outputs.numpy()

    saliency = np.squeeze(output).astype(np.float32)
    # back to the padded input grid, padding cropped away, then to the size of the input image
    if saliency.shape != (_INPUT_H, _INPUT_W):
        saliency = cv2.resize(saliency, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
    saliency = saliency[top:top + new_h, left:left + new_w]
    saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
    saliency = _normalize(saliency)
    return saliency
