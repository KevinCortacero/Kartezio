
import numpy as np
from kartezio.vision.watershed import watershed_transform
from scipy.ndimage import  label



def _extract_markers_dt(signal,threshold):
    distance = signal.copy()  # / signal.max()
    distance[distance < threshold] = 0
    if distance.max() == 0:
        return np.zeros_like(signal)
    markers = label(distance)[0]
    return markers


def watershed_3d(cube,threshold,watershed_line):

    markers = _extract_markers_dt(cube,threshold)

    return watershed_transform(cube,markers,watershed_line)