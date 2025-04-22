import cv2
import numpy as np

from kartezio.vision.rescale import pyrdown


def hough_circles(
    image,
    min_dist=20,
    param1=50,
    param2=30,
    min_radius=0,
    max_radius=0,
    downscale=0,
):
    """
    Find circles in an image using the Hough transform with a downscaled image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    downscale : int
        Number of times to downscale the image.
    min_dist : int, optional
        Minimum distance between the centers of the detected circles (default is 20).
    param1 : int, optional
        First method-specific parameter (default is 50).
    param2 : int, optional
        Second method-specific parameter (default is 30).
    min_radius : int, optional
        Minimum circle radius (default is 0).
    max_radius : int, optional
        Maximum circle radius (default is 0).

    Returns
    -------
    np.ndarray
        The detected circles.
    """
    scale = 2**downscale
    downsampled = pyrdown(image, downscale)
    circles = hough_circles(
        downsampled,
        min_dist=min_dist / scale,
        param1=param1,
        param2=param2,
        min_radius=min_radius / scale,
        max_radius=max_radius / scale,
    )
    circles *= scale
    return circles


def circles_to_mask(image, circles):
    """
    Create a mask from detected circles.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    circles : np.ndarray
        Detected circles.

    Returns
    -------
    np.ndarray
        The mask.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for x, y, r in circles:
        cv2.circle(mask, (x, y), r, 255, -1)
    return mask


def circles_to_labels(image, circles):
    """
    Create a label image from detected circles.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    circles : np.ndarray
        Detected circles.

    Returns
    -------
    np.ndarray
        The label image.
    """
    label = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, (x, y, r) in enumerate(circles, start=1):
        cv2.circle(label, (x, y), r, i, -1)
    return label
