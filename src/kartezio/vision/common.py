import cv2
import numpy as np
from skimage import color


def rgb2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def bgr2hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hsv2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def rgb2hed(image):
    return (color.rgb2hed(image) * 255).astype(np.uint8)


def bgr2hed(image):
    return rgb2hed(bgr2rgb(image))


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
    assert len(image.shape) == 3 or len(image.shape) == 2, (
        "given image wrong format, shape must be (h, w, c) or (h, w)"
    )
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


def fill_polyhedron_as_labels(mask, labels, z_slice, contours):
    for label, slice, polygon in zip(labels, z_slice, contours):
        cv2.fillPoly(mask[slice], pts=np.int32([polygon]), color=label)
    return mask


def contours_as_labels_and_foreground(mask, contours):
    for i, contour in enumerate(contours):
        cv2.fillPoly(mask, pts=np.int32([contour]), color=1)
        cv2.polylines(
            mask, pts=np.int32([contour]), isClosed=True, color=2, thickness=1
        )
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # mask = cv2.GaussianBlur(mask,(3,3),0)
    return mask  # cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((31,31)))
