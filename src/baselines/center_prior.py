import numpy as np


def center_prior_saliency(image_shape, sigma_frac=0.25):
    height, width = image_shape
    #center of image (central row, central column)
    cx = width / 2.0
    cy = height / 2.0
    #sigma of gaussian not constant for different images sizes
    sigma_x = sigma_frac * width
    sigma_y = sigma_frac * height

    #array numpy number 32bits 
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # gaussian function
    exponent = -((xx - cx) ** 2 / (2.0 * sigma_x ** 2)
                 + (yy - cy) ** 2 / (2.0 * sigma_y ** 2))
    gauss = np.exp(exponent)

    # normalize the gaussian to [0, 1]
    return _normalize(gauss)


def _normalize(saliency_map):
    #Min-max normalize a saliency map to [0, 1] and cast it to float32.
    minimum, maximum = saliency_map.min(), saliency_map.max()

    # Rescale so that the minimum maps to 0 and the maximum to 1.
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
   
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)