import cv2
import numpy as np
from scipy import ndimage
from skimage import color
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def image_new(shape, dtype=np.uint8):
    return np.zeros(shape=shape, dtype=dtype)


def image_like(image):
    return np.zeros_like(image)


def image_split(image):
    return list(cv2.split(image))


def image_ew_mean(image_1, image_2):
    return cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)


def image_ew_max(image_1, image_2):
    return cv2.max(image_1, image_2)


def image_ew_min(image_1, image_2):
    return cv2.min(image_1, image_2)


def image_normalize(image):
    return cv2.normalize(
        image,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )


def gradient_magnitude(gx, gy):
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def contours_convex_hull(contour):
    return cv2.convexHull(contour)


def contours_merge(contours):
    return np.vstack(contours)


def contour_area(contour):
    return cv2.contourArea(contour)


def contours_find(image, exclude_holes=True):
    mode = cv2.RETR_LIST
    method = cv2.CHAIN_APPROX_SIMPLE
    if exclude_holes:
        mode = cv2.RETR_EXTERNAL
    return cv2.findContours(image.copy(), mode, method)[0]


def _draw_contours(
    image, contours, color=None, selected: int = None, fill=True, thickness=1
):
    assert (
        len(image.shape) == 3 or len(image.shape) == 2
    ), "given image wrong format, shape must be (h, w, c) or (h, w)"
    if color is None:
        if len(image.shape) == 3 and image.shape[-1] == 3:
            color = [255, 255, 255]
        elif len(image.shape) == 2:
            color = [255]
        else:
            raise ValueError("Image wrong format, must have 1 or 3 channels")
    if selected is None:
        selected = -1  # selects all the contours (-1)
    if fill:
        thickness = -1  # fills the contours (-1)
    return cv2.drawContours(image.copy(), contours, selected, color, thickness)


def contours_fill(image, contours, color=None, selected=None):
    return _draw_contours(image, contours, color, selected, fill=True)


def contours_draw(image, contours, color=None, selected=None, thickness=1):
    return _draw_contours(
        image, contours, color, selected, fill=False, thickness=thickness
    )


class WatershedSkimage:
    def __init__(self, use_dt=False, markers_distance=21, markers_area=None):
        self.use_dt = use_dt
        self.markers_distance = markers_distance
        self.markers_area = markers_area

    def _extract_markers(self, signal):
        peak_idx = peak_local_max(
            signal,
            min_distance=self.markers_distance,
            exclude_border=0,
            # footprint=np.ones((21, 21)),
            num_peaks=255,
        )
        peak_mask = np.zeros_like(signal, dtype=np.uint8)
        peak_mask[tuple(peak_idx.T)] = 1
        return peak_mask

    def apply(self, signal, markers=None, mask=None):
        if self.use_dt:
            signal = ndimage.distance_transform_edt(signal)
        if markers is None:
            markers = self._extract_markers(signal)

        # markers[mask == 0] = 0
        if self.markers_area:
            n, marker_labels, stats, _ = cv2.connectedComponentsWithStats(
                markers, connectivity=8
            )
            for i in range(1, n):
                if (
                    self.markers_area[0]
                    < stats[i, cv2.CC_STAT_AREA]
                    < self.markers_area[1]
                ):
                    pass
                else:
                    marker_labels[marker_labels == i] = 0
        else:
            marker_labels = cv2.connectedComponents(
                markers, connectivity=8, ltype=cv2.CV_16U
            )[1]

        # limit the number of markers to 255
        marker_labels[marker_labels > 255] = 0
        signal_inv = 255 - signal
        labels = watershed(
            signal_inv, markers=marker_labels, mask=mask, watershed_line=True
        )
        return signal, marker_labels, labels


def threshold_binary(image, threshold, value=255):
    return cv2.threshold(image, threshold, value, cv2.THRESH_BINARY)[1]


def threshold_tozero(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Applies a threshold to the input image and sets the pixel values below the threshold to zero.

    Args:
        image (numpy.ndarray): The input image.
        threshold (float): The threshold value.

    Returns:
        numpy.ndarray: The thresholded image.
    """
    return cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)[1]


def fill_ellipses_as_labels(mask, ellipses):
    for i, ellipse in enumerate(ellipses):
        cv2.ellipse(mask, ellipse, i + 1, thickness=-1)
    return mask


def fill_polygons(mask, polygons, color=255):
    return cv2.fillPoly(mask, pts=polygons, color=color)


def fill_polygons_as_labels(mask, polygons):
    for i, polygon in enumerate(polygons):
        cv2.fillPoly(mask, pts=np.int32([polygon]), color=i + 1)
    return mask


def morph_fill(image):
    cnts = contours_find(image, exclude_holes=True)
    return contours_fill(image, cnts)


def draw_overlay(image, mask, color=None, alpha=1.0, border_color="same", thickness=1):
    if color is None:
        color = [255, 255, 255]
    out = image.copy()
    img_layer = image.copy()
    img_layer[np.where(mask)] = color
    overlayed = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)

    n_labels = np.max(mask)
    # compute contours for each label
    for i in range(1, n_labels + 1):
        mask_i = mask == i
        contours = contours_find(mask_i.astype(np.uint8), exclude_holes=False)
        if border_color == "same":
            overlayed = contours_draw(
                overlayed, contours, thickness=thickness, color=color
            )
        elif border_color is not None:
            overlayed = contours_draw(
                overlayed, contours, thickness=thickness, color=border_color
            )
    return overlayed
