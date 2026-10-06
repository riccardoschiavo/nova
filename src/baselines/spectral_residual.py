import numpy as np
import cv2


def spectral_residual_saliency(image, sigma=3.0, target_size=(64, 64)):
  
    original_h, original_w = image.shape[:2]

    # gray conversion for frequency analysis
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # resize to a smaller size for faster computation
    small = cv2.resize(gray, target_size,
                       interpolation=cv2.INTER_AREA).astype(np.float32)

    # frequency analysis: FFT, amplitude and phase
    # fft2 computes 2d discrete f. transform
    spectrum = np.fft.fft2(small)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    
    log_amplitude = np.log(amplitude + 1e-8) # value compression
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3)) # local mean computation
    spectral_residual = log_amplitude - avg_log_amplitude # subtraction -> peaks

    # find the real and imaginary values
    # fft.ifft2 reconstruts the image
    reconstructed = np.fft.ifft2(np.exp(spectral_residual + 1j * phase))
    # magnitude squared for increased contrast
    saliency = (np.abs(reconstructed) ** 2).astype(np.float32) 

    # Kernel size 6*sigma 
    # | 1 makes it odd (required by OpenCV central pixel)
    ksize = int(np.ceil(sigma * 6)) | 1
    saliency = cv2.GaussianBlur(saliency, (ksize, ksize), sigma) # smoothing as the human gaze 

    saliency = _normalize(saliency)
    #return to original size
    return cv2.resize(saliency, (original_w, original_h),
                      interpolation=cv2.INTER_LINEAR)

# normalizarion function to [0, 1] range
def _normalize(saliency_map):
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)
