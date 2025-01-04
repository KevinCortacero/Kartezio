from abc import ABC
from typing import Dict

import cv2
import numpy as np
from skimage.transform import hough_ellipse

from kartezio.core.components import Endpoint, registry

# from kartezio.preprocessing import Resize
# from kartezio.primitives.array import Sobel
from kartezio.types import TypeArray, TypeLabels
from kartezio.vision.common import (
    contours_fill,
    contours_find,
    image_new,
    threshold_tozero,
)
from kartezio.vision.watershed import (
    _connected_components,
    distance_watershed,
    double_threshold_watershed,
    local_max_watershed,
    marker_controlled_watershed,
    threshold_local_max_watershed,
    threshold_watershed,
)


class EndpointWatershed(Endpoint, ABC):
    def __init__(self, arity, watershed_line=True):
        super().__init__([TypeArray] * arity)
        self.watershed_line = watershed_line


class PeakedMarkersWatershed(EndpointWatershed, ABC):
    def __init__(self, watershed_line=True, min_distance=1, downsample=0):
        super().__init__(1, watershed_line=watershed_line)
        self.min_distance = min_distance
        self.downsample = downsample


@registry.add_as(Endpoint)
class MarkerControlledWatershed(EndpointWatershed):
    """
    MarkerControlledWatershed

    An endpoint class for a Cartesian Genetic Programming pipeline (or other plugin system)
    that applies a marker-controlled watershed algorithm to segment an image.

    The input is expected to be a list/tuple of two NumPy arrays:
        1) `x[0]`: The primary image (e.g., grayscale or distance-transformed) to be segmented.
        2) `x[1]`: The binary or labeled marker array specifying the regions or points
                   from which the watershed should grow.

    Parameters
    ----------
    watershed_line : bool, optional
        If True, the watershed algorithm computes a "line" (pixel value = 0)
        that separates adjacent segmented regions. Defaults to True.

    Examples
    --------
    >>> # Suppose 'img' is a 2D NumPy array (grayscale)
    >>> # and 'markers' is a 2D array of the same shape, with nonzero regions.
    >>> segmenter = MarkerControlledWatershed(watershed_line=False)
    >>> segmented = segmenter.call([img, markers])[0]
    >>> segmented.shape
    (height, width)
    """

    def __init__(self, watershed_line=True):
        super().__init__(2, watershed_line=watershed_line)

    def call(self, x):
        """
        Apply the marker-controlled watershed transform to the given input and marker arrays.

        Parameters
        ----------
        x : list or tuple of np.ndarray
            A container with two arrays:
              - x[0]: The primary image (2D ndarray) to be segmented.
              - x[1]: The marker array (2D ndarray) identifying regions/seeds.

        Returns
        -------
        list of np.ndarray
            A single-element list containing the segmented (labeled) image as a 2D ndarray.
            Each connected region is assigned a unique integer label.
        """
        return [marker_controlled_watershed(x[0], x[1], self.watershed_line)]


@registry.add_as(Endpoint)
class LocalMaxWatershed(PeakedMarkersWatershed):
    """
    LocalMaxWatershed

    An endpoint that finds local maxima within a single input image (which may
    be a grayscale or distance-transformed image) to generate markers, then
    applies the watershed transform.

    Parameters
    ----------
    min_distance : int, optional
        The minimum distance separating local maxima. Defaults to 10.
    watershed_line : bool, optional
        If True, produces watershed lines (pixel = 0) where regions meet.
        Defaults to True.
    downsample : int, optional
        If > 0, downsample the input by 2^downsample before detecting maxima.
        Defaults to 0 (no downsampling).
    """

    def __init__(
        self, watershed_line: bool, min_distance: int, downsample: int = 0
    ):
        super().__init__(
            watershed_line, min_distance, downsample
        )  # Single input image

    def call(self, x):
        """
        Perform local-maxima-based watershed on the input image.

        Parameters
        ----------
        x : list of np.ndarray
            - x[0] : The image on which to perform watershed (2D ndarray).

        Returns
        -------
        list of np.ndarray
            A single-element list containing the labeled segmentation result.
        """
        return [
            local_max_watershed(
                image=x[0],
                min_distance=self.min_distance,
                watershed_line=self.watershed_line,
                downsample=self.downsample,
            )
        ]


@registry.add_as(Endpoint)
class DistanceWatershed(PeakedMarkersWatershed):
    """
    DistanceWatershed

    An endpoint that computes a distance transform internally, finds local maxima,
    and applies a watershed transform. Typically used for segmenting binary masks
    (foreground vs. background).

    Parameters
    ----------
    min_distance : int, optional
        Minimum distance to separate local maxima in the distance map. Defaults to 10.
    watershed_line : bool, optional
        Whether to produce watershed lines. Defaults to True.
    downsample : int, optional
        If > 0, downsample the distance-transformed image before local maxima detection.
        Defaults to 0.
    """

    def __init__(
        self, watershed_line: bool, min_distance: int, downsample: int = 0
    ):
        super().__init__(
            watershed_line, min_distance, downsample
        )  # Single input (binary mask recommended)

    def call(self, x):
        """
        Perform distance-based watershed on a binary mask or grayscale image.

        Parameters
        ----------
        x : list of np.ndarray
            - x[0] : The input image (2D ndarray), typically a binary mask.

        Returns
        -------
        list of np.ndarray
            A single-element list containing the labeled segmentation.
        """
        return [
            distance_watershed(
                image=x[0],
                min_distance=self.min_distance,
                watershed_line=self.watershed_line,
                downsample=self.downsample,
            )
        ]


@registry.add_as(Endpoint)
class ThresholdLocalMaxWatershed(PeakedMarkersWatershed):
    """
    ThresholdLocalMaxWatershed

    An endpoint that first thresholds the input image (zeroing out pixels below
    the threshold), then detects local maxima in the thresholded image and applies
    a watershed transform.

    Parameters
    ----------
    threshold : float, optional
        Pixel intensity threshold. Pixels below are zeroed out. Defaults to 128.0.
    min_distance : int, optional
        Minimum distance separating local maxima. Defaults to 10.
    watershed_line : bool, optional
        If True, produce watershed lines. Defaults to True.
    downsample : int, optional
        If > 0, downsample before local maxima detection. Defaults to 0.
    """

    def __init__(
        self,
        watershed_line: bool = True,
        min_distance: int = 10,
        downsample: int = 0,
        threshold: int = 128,
    ):
        super().__init__(
            watershed_line, min_distance, downsample
        )  # Single input image
        self.threshold = threshold
        self.min_distance = min_distance
        self.watershed_line = watershed_line
        self.downsample = downsample

    def call(self, x):
        """
        Apply threshold + local maxima + watershed to the input image.

        Parameters
        ----------
        x : list of np.ndarray
            - x[0] : The input image (2D ndarray) to be thresholded & segmented.

        Returns
        -------
        list of np.ndarray
            A single-element list containing the labeled segmentation result.
        """
        return [
            threshold_local_max_watershed(
                image=x[0],
                threshold=self.threshold,
                min_distance=self.min_distance,
                watershed_line=self.watershed_line,
                downsample=self.downsample,
            )
        ]


@registry.add_as(Endpoint)
class ThresholdWatershed(EndpointWatershed):
    """
    ThresholdWatershed

    An endpoint that applies a threshold to the input image to generate a marker
    array, then directly performs a marker-controlled watershed.

    Parameters
    ----------
    threshold : float, optional
        Pixel intensity threshold below which pixels become 0 in the marker array.
        Defaults to 128.0.
    watershed_line : bool, optional
        If True, produce watershed lines. Defaults to True.
    """

    def __init__(
        self, watershed_line: bool, threshold: int = 128, threshold_2=None
    ):
        super().__init__(1, watershed_line)  # Single input image
        self.threshold = threshold
        self.threshold_2 = threshold_2
        if threshold_2 is not None:
            if not (self.threshold < self.threshold_2):
                raise ValueError(
                    f"threshold1 ({self.threshold}) must be < threshold2 ({self.threshold_2})"
                )

    def call(self, x):
        """
        Apply threshold-based watershed to the input image.

        Parameters
        ----------
        x : list of np.ndarray
            - x[0] : The input image (2D ndarray).

        Returns
        -------
        list of np.ndarray
            A single-element list containing the labeled segmentation result.
        """
        if self.threshold_2 is not None:
            return [
                double_threshold_watershed(
                    image=x[0],
                    threshold1=self.threshold1,
                    threshold2=self.threshold2,
                    watershed_line=self.watershed_line,
                )
            ]
        return [
            threshold_watershed(
                image=x[0],
                threshold=self.threshold,
                watershed_line=self.watershed_line,
            )
        ]


@registry.add_as(Endpoint)
class ToLabels(Endpoint):
    def __init__(self):
        super().__init__([TypeArray])

    def call(self, x):
        return [_connected_components(x[0])]


@registry.add_as(Endpoint)
class EndpointSubtract(Endpoint):
    def __init__(self):
        super().__init__([TypeArray, TypeArray])

    def call(self, x):
        return [cv2.subtract(x[0], x[1])]


@registry.add_as(Endpoint)
class EndpointThreshold(Endpoint):
    def __init__(self, threshold):
        super().__init__([TypeArray])
        self.threshold = threshold

    def call(self, x):
        return [threshold_tozero(x[0], self.threshold)]

    def __to_dict__(self) -> Dict:
        return {
            "args": {
                "threshold": self.threshold,
            },
        }


print(registry.display())
test_endpoint = Endpoint.from_config(
    {
        "name": "ThresholdWatershed",
        "args": {"watershed_line": True, "threshold": 128},
    }
)
print(test_endpoint)
print(test_endpoint.__to_dict__())


@register(Endpoint)
class EndpointHoughCircle(Endpoint):
    def __init__(
        self, min_dist=21, p1=128, p2=64, min_radius=20, max_radius=120
    ):
        super().__init__([TypeArray])
        self.min_dist = min_dist
        self.p1 = p1
        self.p2 = p2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def call(self, x):
        mask = x[0]
        n = 0
        new_mask = image_new(mask.shape)
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            1,
            self.min_dist,
            param1=self.p1,
            param2=self.p2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, circle in enumerate(circles[0, :]):
                center = (circle[0], circle[1])
                # circle outline
                radius = circle[2]
                cv2.circle(new_mask, center, radius, (i + 1), -1)
                n += 1

        return [new_mask]


@register(Endpoint)
class EndpointEllipse(Endpoint):
    def __init__(
        self,
        min_axis=10,
        max_axis=30,
        min_ratio=0.5,
        backend="opencv",
        keep_mask=True,
        as_labels=False,
    ):
        super().__init__([TypeArray])
        self.min_axis = min_axis
        self.max_axis = max_axis
        self.min_ratio = min_ratio
        self.backend = backend
        self.keep_mask = keep_mask
        self.as_labels = as_labels

    def call(self, x, args=None):
        mask = x[0]
        n = 0
        new_labels = image_new(mask.shape)
        new_mask = image_new(mask.shape)
        labels = []
        # edges = self.edge_detector.call([mask], [3, 3])
        if self.backend == "opencv":
            cnts = contours_find(x[0], exclude_holes=True)
            for cnt in cnts:
                if len(cnt) >= 5:
                    center, (MA, ma), angle = cv2.fitEllipse(cnt)
                    if (
                        self.min_axis <= MA <= self.max_axis
                        and self.min_axis <= ma <= self.max_axis
                        and ma / MA >= self.min_ratio
                    ):
                        if self.keep_mask:
                            new_mask = contours_fill(new_mask, [cnt], n + 1)
                        else:
                            cv2.ellipse(
                                new_mask,
                                (center, (MA, ma), angle),
                                n + 1,
                                thickness=-1,
                            )
                            labels.append((center, (MA, ma), angle))
                        n += 1
            if self.keep_mask:
                if self.as_labels:
                    new_mask[new_mask >= 1] = mask[new_mask >= 1]
                else:
                    new_mask[new_mask >= 1] = x[0][new_mask >= 1]

        elif self.backend == "skimage":
            print("skimage")

            print("edges", edges)
            ellipses = hough_ellipse(
                edges,
                threshold=128,
                min_size=self.min_axis,
                max_size=self.max_axis,
            )
            for ellipse in ellipses:
                print(ellipse)
                _, y0, x0, a, b, o = ellipse
                cv2.ellipse(
                    new_labels,
                    ((x0, y0), (a, b), o),
                    n + 1,
                    thickness=-1,
                )
                labels.append((x0, y0, a, b))
                n += 1

        return [new_mask]

    def __to_dict__(self) -> Dict:
        return {
            "name": "fit_ellipse",
            "args": {
                "min_axis": self.min_axis,
                "max_axis": self.max_axis,
                "min_ratio": self.min_ratio,
                "backend": self.backend,
                "keep_mask": self.keep_mask,
                "as_labels": self.as_labels,
            },
        }


@register(Endpoint)
class EndpointHoughCircleSmall(Endpoint):
    def __init__(self, min_dist=4, p1=256, p2=8, min_radius=2, max_radius=12):
        super().__init__([TypeArray])
        self.min_dist = min_dist
        self.p1 = p1
        self.p2 = p2
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.edge_detector = Sobel()

    def _to_json_kwargs(self) -> dict:
        return {
            "min_dist": self.min_dist,
            "p1": self.p1,
            "p2": self.p2,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
        }

    def call(self, x):
        mask_raw = x[0]
        new_mask = image_new(mask_raw.shape)
        mask = self.edge_detector.call([mask_raw], [3, 3])
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            1,
            self.min_dist,
            param1=self.p1,
            param2=self.p2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i, circle in enumerate(circles[0, :]):
                center = (circle[0], circle[1])
                # circle outline
                radius = circle[2]
                cv2.circle(new_mask, center, radius, (i + 1), -1)
        return [new_mask]


@register(Endpoint)
class EndpointRescale(Endpoint):
    def __init__(self, scale, method):
        super().__init__([TypeArray])
        self.resize = Resize(scale, method)

    def call(self, x):
        return self.resize.call([x])[0]
