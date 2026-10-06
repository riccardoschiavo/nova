import numpy as np
import cv2


def itti_koch_saliency(image, pyramid_levels=7):
    # --- Feature extraction -------------------------------------------------
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)

    
    intensity = (r + g + b) / 3.0 #brightness channel

    eps = 1e-3 # small constant to avoid division by zero
    # compute two colour-opponent channels: red-green and blue-yellow
    rg = (r - g) / (intensity + eps) 
    by = (b - np.minimum(r, g)) / (intensity + eps)

    # Look at multiple resolution levels (Gaussian pyramid) for each channel
    intensity_pyr = _gaussian_pyramid(intensity, pyramid_levels)
    rg_pyr = _gaussian_pyramid(rg, pyramid_levels)
    by_pyr = _gaussian_pyramid(by, pyramid_levels)

    # Gabor filter for orientation: 0, 45, 90, 135 degrees
    orientation_pyramids = []
    for theta_deg in (0, 45, 90, 135):
        # Create the Gabor kernel for the current orientation
        kernel = cv2.getGaborKernel(
            ksize=(15, 15), sigma=4.0, theta=np.radians(theta_deg),
            lambd=10.0, gamma=0.5, psi=0,
        )
        # slide the kernel over each level of the intensity pyramid
        orientation_pyramids.append(
            [cv2.filter2D(level, cv2.CV_32F, kernel) for level in intensity_pyr]
        )

    # --- Weighted maps split by feature category ----------------------------
    intensity_maps = [_normalize_map(m) for m in _center_surround(intensity_pyr)]

    color_maps = ([_normalize_map(m) for m in _center_surround(rg_pyr)]
                  + [_normalize_map(m) for m in _center_surround(by_pyr)])

    orientation_maps = []
    for pyr in orientation_pyramids:
        orientation_maps += [_normalize_map(m) for m in _center_surround(pyr)]

    # --- Conspicuity maps ---------------------------------------------------

    # target shape for resizing    
    target_shape = intensity_maps[0].shape

    def combine(maps):
        resized = [_resize_to(m, target_shape) for m in maps]
        return _normalize_map(np.sum(resized, axis=0)) # sum and normalization

    # combine maps of the same feature type into a single map
    conspicuity_intensity = combine(intensity_maps)
    conspicuity_color = combine(color_maps)
    conspicuity_orientation = combine(orientation_maps)

    # --- Final saliency map -------------------------------------------------
    
    # Average of the 3 resulting maps
    saliency = (conspicuity_intensity
                + conspicuity_color
                + conspicuity_orientation) / 3.0

    # Resize to original image size
    saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

   # Normalization to [0, 1]
    minimum, maximum = saliency.min(), saliency.max()
    if maximum - minimum > 0:
        saliency = (saliency - minimum) / (maximum - minimum)
    else:
        saliency = np.zeros_like(saliency)
    return saliency.astype(np.float32)


def _gaussian_pyramid(channel, levels):
    # Create a Gaussian pyramid for a single channel
    pyramid = [channel.astype(np.float32)]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]).astype(np.float32))
    return pyramid


def _resize_to(feature_map, shape):
    # Resize a feature map to a target
    return cv2.resize(feature_map, (shape[1], shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def _center_surround(pyramid, center_scales=(2, 3), deltas=(3, 4)):
    
    # A pixel is salient if it differs from the average of its surroundings.
    # The coarse (blurred) level approximates that surrounding average.    
    # Result: a list of centre-surround difference maps.
    
    maps = []
    for c in center_scales:
        for delta in deltas:
            s = c + delta
            if s >= len(pyramid):
                continue
            center = pyramid[c]
            surround = _resize_to(pyramid[s], center.shape[:2])
            maps.append(np.abs(center - surround))
    return maps


def _normalize_map(feature_map, M=10.0):
   
    # Normalize the map to the fixed range until M
    feature_map = feature_map - feature_map.min()
    if feature_map.max() > 0:
        feature_map = feature_map / feature_map.max() * M

   # find local peak 
    local_max = cv2.dilate(feature_map, np.ones((7, 7), np.uint8))
    is_local_max = (feature_map == local_max) & (feature_map < M - 1e-3)
    peaks = feature_map[is_local_max]

    # compute the mean of the local peaks
    mean_peak = peaks.mean() if peaks.size > 0 else 0.0

    #if the mean of the peaks is high the map is noisy, the exponential will be lower and will penalize the map
    #if the mean is low the map is clean, the exponential will be higher and will reward the map
    return feature_map * (M - mean_peak) ** 2