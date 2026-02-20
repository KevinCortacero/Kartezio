"""
watershed.py

This module provides a collection of image segmentation routines based on the
watershed transform. It relies on OpenCV (cv2) for image processing operations
(e.g., distance transforms, connected components) and on scikit-image for
finding local maxima (peak_local_max) and performing the watershed algorithm.

Functions Overview
------------------
- watershed_transform(image, markers, watershed_line):
    Applies the watershed transform using negative-exponential-scaled intensity.

- _connected_components(image):
    Labels connected components of a binary image using cv2.connectedComponents.

- _distance_transform(image, normalize=False):
    Computes the distance transform of a binary image; optionally normalizes it.

- coordinates_to_mask(coordinates, shape, scale):
    Converts peak coordinates into a binary mask, optionally rescaling them.

- _local_max_markers(image, min_distance):
    Creates markers for watershed by identifying local maxima in an image.

- _fast_local_max_markers(image, min_distance, n):
    Performs local maxima detection on a downsampled image, then scales results.

- marker_controlled_watershed(image, markers, watershed_line):
    Runs a watershed transform using the given marker image.

- local_max_watershed(image, min_distance, watershed_line, downsample):
    Combines a distance transform or intensity-based approach with local maxima
    to produce markers and apply watershed.

- distance_watershed(image, min_distance, watershed_line, downsample=0):
    Shortcut to compute a distance transform, detect local maxima, and run watershed.

- threshold_local_max_watershed(image, threshold, min_distance, watershed_line, downsample=0):
    Applies a threshold, finds local maxima, and runs watershed with those markers.

- threshold_watershed(image, threshold, watershed_line):
    Applies a threshold directly as markers for watershed.

- double_threshold_watershed(image, threshold1, threshold2, watershed_line):
    Uses two thresholds to create marker regions for watershed.

Dependencies
------------
- cv2 (OpenCV)
- numpy
- skimage.feature.peak_local_max
- skimage.segmentation.watershed
"""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from kartezio.vision.common import threshold_tozero
from kartezio.vision.rescale import pyrdown


def watershed_transform(image, markers, watershed_line):
    """
    Apply a watershed transform using negative exponential scaling of intensities.

    Parameters
    ----------
    image : np.ndarray
        Input grayscale image. Expected range [0, 255] for scaling.
    markers : np.ndarray
        Marker image, typically an integer-labeled array where non-zero regions
        represent different labels.
    watershed_line : bool
        If True, the function computes a lines-producing watershed. The lines
        separate the regions, setting them to 0.

    Returns
    -------
    np.ndarray
        Labeled image (same shape as input). Each connected region has its own label.
        Watershed lines are zero if watershed_line=True.
    """
    scaled = cv2.exp((image / 255.0).astype(np.float32))
    return watershed(
        -scaled,
        markers=markers,
        mask=image > 0,
        watershed_line=watershed_line,
    )


def _connected_components(image):
    """
    Label the connected components in a binary image.

    Parameters
    ----------
    image : np.ndarray
        Binary or label image from which connected components are computed.

    Returns
    -------
    np.ndarray
        An integer-labeled array of the same shape, where each component has a unique ID.
    """
    # cv2.connectedComponents returns (num_labels, labels_img). We only return labels_img.
    return cv2.connectedComponents(image, connectivity=8, ltype=cv2.CV_16U)[1]


def _distance_transform(image, normalize=False):
    """
    Compute the distance transform of a binary image.

    Parameters
    ----------
    image : np.ndarray
        Binary image (non-zero pixels considered foreground).
    normalize : bool, optional
        If True, the resulting distance transform is normalized to the range [0,1].

    Returns
    -------
    np.ndarray
        A distance transform of the same shape as `image`.
    """
    distance = cv2.distanceTransform(image, cv2.DIST_L2, 3)
    if normalize:
        return cv2.normalize(distance, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return distance


def coordinates_to_mask(coordinates, shape, scale):
    """
    Convert coordinates of local maxima into a binary mask.

    Parameters
    ----------
    coordinates : np.ndarray
        An (N, 2) array of (row, col) coordinates for local maxima.
    shape : tuple
        Shape of the desired mask (e.g., image.shape).
    scale : int
        Upsampling factor to rescale coordinates if they were found in a downsampled image.

    Returns
    -------
    np.ndarray
        Binary mask of the same shape with 1's at local maxima positions, 0 otherwise.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    coordinates = (coordinates * scale).astype(np.int32)
    mask[coordinates[:, 0], coordinates[:, 1]] = 255
    return mask


def _local_max_markers(image, min_distance):
    """
    Identify local maxima in the image and convert them to a binary marker mask.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or distance-transformed image.
    min_distance : int
        Minimum distance separating local maxima.

    Returns
    -------
    np.ndarray
        Binary mask of local maxima.
    """
    peak_idx = peak_local_max(
        image,
        min_distance=min_distance,
        exclude_border=1,
    )
    return coordinates_to_mask(peak_idx, image.shape, 1)


def _fast_local_max_markers(image, min_distance, n):
    """
    Identify local maxima on a downsampled image, then scale them back up.

    Parameters
    ----------
    image : np.ndarray
        Input image to find local maxima.
    min_distance : int
        Minimum distance separating local maxima (adjusted for downsampling).
    n : int
        Number of times the image is downsampled.

    Returns
    -------
    np.ndarray
        Binary mask of local maxima upsampled to match the original image size.
    """
    scale = 2**n
    image_down = pyrdown(image, n)
    peak_coordinates = peak_local_max(
        image_down,
        min_distance=min_distance // scale,
        exclude_border=1,
    )
    return coordinates_to_mask(peak_coordinates, image.shape, scale)


def marker_controlled_watershed(image, markers, watershed_line):
    """
    Run a watershed transform given an initial marker image.

    Parameters
    ----------
    image : np.ndarray
        Input image to be segmented (e.g., intensity or distance transform).
    markers : np.ndarray
        Binary or labeled image to serve as markers.
    watershed_line : bool
        If True, produce watershed lines (separating boundaries).

    Returns
    -------
    np.ndarray
        Integer-labeled segmented image.
    """
    markers = _connected_components(markers)
    return watershed_transform(image, markers, watershed_line)


def local_max_watershed(image, min_distance, watershed_line, downsample):
    """
    Segment an image by detecting local maxima as markers and running watershed.

    Parameters
    ----------
    image : np.ndarray
        Image to segment (could be a distance transform or raw intensity).
    min_distance : int
        Minimum distance separating local maxima.
    watershed_line : bool
        Whether to produce watershed lines.
    downsample : int or None
        If > 0, downsample the image by 'downsample' times before detecting maxima.
        Otherwise, detect maxima at the full resolution.

    Returns
    -------
    np.ndarray
        Integer-labeled segmented image.
    """
    if downsample:
        markers = _fast_local_max_markers(image, min_distance, downsample)
    else:
        markers = _local_max_markers(image, min_distance)
    return marker_controlled_watershed(image, markers, watershed_line)


def distance_watershed(image, min_distance, watershed_line, normalize, downsample=0):
    """
    Shortcut for running watershed using a distance transform + local maxima approach.

    Parameters
    ----------
    image : np.ndarray
        Binary image on which distance transform is computed.
    min_distance : int
        Minimum distance for local maxima detection.
    watershed_line : bool
        Whether to produce watershed lines.
    downsample : int, optional
        If > 0, downsample the distance transform before local maxima detection.

    Returns
    -------
    np.ndarray
        Integer-labeled segmentation of the input image.
    """
    distance = _distance_transform(image, normalize=normalize)
    return local_max_watershed(distance, min_distance, watershed_line, downsample)


def threshold_local_max_watershed(
    image, threshold, min_distance, watershed_line, downsample=0
):
    """
    Watershed segmentation where markers are found from thresholded local maxima.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or distance-transformed image.
    threshold : float
        Pixel intensity threshold; below this, pixels become 0.
    min_distance : int
        Minimum distance for local maxima detection.
    watershed_line : bool
        Whether to produce watershed lines.
    downsample : int, optional
        If > 0, downsample before local maxima detection.

    Returns
    -------
    np.ndarray
        Segmented, labeled image.
    """
    markers = threshold_tozero(image, threshold)
    if downsample:
        markers = _fast_local_max_markers(markers, min_distance, downsample)
    else:
        markers = _local_max_markers(markers, min_distance)
    return marker_controlled_watershed(image, markers, watershed_line)


def threshold_watershed(image, threshold, watershed_line):
    """
    Basic threshold-based watershed segmentation.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or distance-transformed image to segment.
    threshold : float
        Pixel intensity threshold; below this, pixels become 0 (markers).
    watershed_line : bool
        Whether to produce watershed lines.

    Returns
    -------
    np.ndarray
        Labeled watershed segmentation.
    """
    markers = threshold_tozero(image, threshold)
    return marker_controlled_watershed(image, markers, watershed_line)


def double_threshold_watershed(image, threshold1, threshold2, watershed_line):
    """
    Watershed segmentation using two thresholds.

    Pixels below threshold1 are set to zero, then below threshold2 are further
    refined. This can help isolate more confident markers from less confident ones.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or distance-transformed image.
    threshold1 : float
        First threshold; pixels below are zeroed out.
    threshold2 : float
        Second threshold; pixels below are zeroed out (refined).
    watershed_line : bool
        Whether to produce watershed lines.

    Returns
    -------
    np.ndarray
        Labeled segmentation from watershed.
    """

    image = threshold_tozero(image, threshold1)
    markers = threshold_tozero(image, threshold2)
    """
    markers = np.zeros_like(image)
    markers[image < threshold1] = 1
    markers[image > threshold2] = 2
    """

    return marker_controlled_watershed(image, markers, watershed_line)
