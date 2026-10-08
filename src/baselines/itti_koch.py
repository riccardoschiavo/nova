import numpy as np
import cv2


def itti_koch_saliency(image, center_scales=(2, 3, 4), deltas=(3, 4),
                       conspicuity_scale=4):
    # --- Feature extraction -------------------------------------------------
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)

    intensity = (r + g + b) / 3.0 #brightness channel

    pyramid_levels = max(center_scales) + max(deltas) + 1

    mask = intensity > 0.1 * intensity.max()
    safe_intensity = np.where(mask, intensity, 1.0)
    r = np.where(mask, r / safe_intensity, 0.0).astype(np.float32)
    g = np.where(mask, g / safe_intensity, 0.0).astype(np.float32)
    b = np.where(mask, b / safe_intensity, 0.0).astype(np.float32)

    red = np.maximum(r - (g + b) / 2.0, 0.0)
    green = np.maximum(g - (r + b) / 2.0, 0.0)
    blue = np.maximum(b - (r + g) / 2.0, 0.0)
    yellow = np.maximum((r + g) / 2.0 - np.abs(r - g) / 2.0 - b, 0.0)

    # Look at multiple resolution levels (Gaussian pyramid) for each channel
    intensity_pyr = _gaussian_pyramid(intensity, pyramid_levels)
    red_pyr = _gaussian_pyramid(red, pyramid_levels)
    green_pyr = _gaussian_pyramid(green, pyramid_levels)
    blue_pyr = _gaussian_pyramid(blue, pyramid_levels)
    yellow_pyr = _gaussian_pyramid(yellow, pyramid_levels)

    rg_pyr = [rr - gg for rr, gg in zip(red_pyr, green_pyr)]
    gr_pyr = [gg - rr for rr, gg in zip(red_pyr, green_pyr)]
    by_pyr = [bb - yy for bb, yy in zip(blue_pyr, yellow_pyr)]
    yb_pyr = [yy - bb for bb, yy in zip(blue_pyr, yellow_pyr)]

    # Gabor filter for orientation: 0, 45, 90, 135 degrees
    orientation_pyramids = []
    for theta_deg in (0, 45, 90, 135):
        # Create the Gabor kernel for the current orientation
        kernel = cv2.getGaborKernel(
            ksize=(15, 15), sigma=4.0, theta=np.radians(theta_deg),
            lambd=10.0, gamma=0.5, psi=0,
        )
        kernel -= kernel.mean()
        # slide the kernel over each level of the intensity pyramid
        orientation_pyramids.append(
            [cv2.filter2D(level, cv2.CV_32F, kernel) for level in intensity_pyr]
        )

    # --- Feature maps -------------------------------------------------------
    def center_surround(center_pyr, surround_pyr):
        return _center_surround(center_pyr, surround_pyr, center_scales, deltas)

    intensity_maps = center_surround(intensity_pyr, intensity_pyr)
    rg_maps = center_surround(rg_pyr, gr_pyr)
    by_maps = center_surround(by_pyr, yb_pyr)

    # --- Conspicuity maps ---------------------------------------------------

    # target shape for resizing
    target_shape = intensity_pyr[conspicuity_scale].shape

    def across_scale_add(maps):
        return np.sum([_resize_to(m, target_shape) for m in maps], axis=0)

    conspicuity_intensity = across_scale_add(
        [_normalize_map(m) for m in intensity_maps])
    conspicuity_color = across_scale_add(
        [_normalize_map(rg) + _normalize_map(by) for rg, by in zip(rg_maps, by_maps)])
    conspicuity_orientation = np.sum([
        _normalize_map(across_scale_add(
            [_normalize_map(m) for m in center_surround(pyr, pyr)]))
        for pyr in orientation_pyramids
    ], axis=0)

    # --- Final saliency map -------------------------------------------------

    # Average of the 3 resulting maps
    saliency = (_normalize_map(conspicuity_intensity)
                + _normalize_map(conspicuity_color)
                + _normalize_map(conspicuity_orientation)) / 3.0

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
    if shape[0] < feature_map.shape[0]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(feature_map.astype(np.float32), (shape[1], shape[0]),
                      interpolation=interpolation)


def _center_surround(center_pyr, surround_pyr, center_scales, deltas):

    # A pixel is salient if it differs from the average of its surroundings.
    # The coarse (blurred) level approximates that surrounding average.
    # Result: a list of centre-surround difference maps.

    maps = []
    for c in center_scales:
        for delta in deltas:
            s = c + delta
            center = center_pyr[c]
            surround = _resize_to(surround_pyr[s], center.shape[:2])
            maps.append(np.abs(center - surround))
    return maps


def _normalize_map(feature_map, M=10.0):

    # Normalize the map to the fixed range until M
    feature_map = (feature_map - feature_map.min()).astype(np.float32)
    if feature_map.max() > 0:
        feature_map = feature_map / feature_map.max() * M

   # find local peak
    neighbourhood = np.ones((7, 7), np.uint8)
    local_max = cv2.dilate(feature_map, neighbourhood)
    local_min = cv2.erode(feature_map, neighbourhood)
    is_local_max = ((feature_map == local_max) & (feature_map > local_min)
                    & (feature_map < M - 1e-3))
    peaks = feature_map[is_local_max]

    # compute the mean of the local peaks
    mean_peak = peaks.mean() if peaks.size > 0 else 0.0

    #if the mean of the peaks is high the map is noisy, the exponential will be lower and will penalize the map
    #if the mean is low the map is clean, the exponential will be higher and will reward the map
    return feature_map * (M - mean_peak) ** 2
