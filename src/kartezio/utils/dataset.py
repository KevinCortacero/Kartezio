import cv2
import numpy as np
from skimage.data import cell


def one_cell_dataset():
    image: np.ndarray = cell()
    train_x = [[image]]
    image_annotation = np.zeros_like(image)
    cv2.circle(image_annotation, (428, 375), 64, 255, -1)
    train_y = [[image_annotation]]
    return train_x, train_y
