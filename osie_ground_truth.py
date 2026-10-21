# OSIE ground truth: reads the observers' fixations and turns them into heatmaps

import numpy as np
import scipy.io as sio

SIGMA_FRAC = 0.05 # gaussian sigma as a fraction of the image width

# reads fixations.mat and returns a dictionary {image name: list of (x, y) points}
def load_fixations(mat_path):

    data = sio.loadmat(mat_path)
    entries = data["fixations"] # one record per image

    result = {}
    for i in range(entries.shape[0]):
        record = entries[i, 0][0, 0] # scipy loads matlab structs as nested arrays
        image_name = str(record["img"][0])

        # merges the fixations of all the observers of the image
        points = []
        subjects = record["subjects"]
        for s in range(subjects.shape[0]):
            subject_data = subjects[s, 0][0, 0]
            xs = np.asarray(subject_data["fix_x"]).ravel() # x coordinates of the fixations
            ys = np.asarray(subject_data["fix_y"]).ravel() # y coordinates of the fixations
            points.extend(zip(xs.tolist(), ys.tolist()))

        result[image_name] = points

    return result


# creates a heatmap by summing a gaussian centered on each fixation point
def points_to_heatmap(points, image_shape, sigma_frac=SIGMA_FRAC):

    height, width = image_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    if not points:
        return heatmap # no fixations: empty map

    sigma = sigma_frac * width # sigma proportional to the image width
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32) # coordinates of every pixel

    for x, y in points:
        squared_distance = (x_grid - x) ** 2 + (y_grid - y) ** 2 # squared distance from the point
        heatmap += np.exp(-squared_distance / (2.0 * sigma ** 2)) # gaussian centered on the point

    # normalizes to [0, 1] by dividing by the maximum
    max_value = heatmap.max()
    if max_value > 0:
        heatmap /= max_value
    return heatmap
