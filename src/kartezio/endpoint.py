from typing import Dict

import cv2
import numpy as np
from kartezio.components.core import register
from kartezio.components.endpoint import Endpoint
from kartezio.libraries.array import Sobel
from kartezio.types import TypeArray, TypeLabels
from kartezio.vision.common import (
    WatershedSkimage,
    contours_fill,
    contours_find,
    image_new,
    threshold_tozero,
)
from kartezio.preprocessing import Resize
from skimage.segmentation import watershed
from skimage.transform import hough_ellipse
from skimage.feature import peak_local_max
from scipy import ndimage


@register(Endpoint, "to_labels")
class ToLabels(Endpoint):
    def call(self, x):
        return [
            x[0],
            cv2.connectedComponents(
                x[0], connectivity=self.connectivity, ltype=cv2.CV_16U
            )[1],
        ]

    def __init__(self, connectivity=4):
        super().__init__([TypeArray])
        self.connectivity = connectivity


@register(Endpoint, "subtract")
class EndpointSubtract(Endpoint):
    def __init__(self):
        super().__init__([TypeArray, TypeArray])

    def call(self, x):
        return [cv2.subtract(x[0], x[1])]


@register(Endpoint, "threshold")
class EndpointThreshold(Endpoint):
    def __init__(self, threshold, normalize=False, mode="binary"):
        super().__init__([TypeArray])
        self.threshold = threshold
        self.normalize = normalize
        self.mode = mode

    def call(self, x):
        image = x[0]
        if self.normalize:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        if self.mode == "binary":
            return [cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)[1]]
        return [cv2.threshold(image, self.threshold, 255, cv2.THRESH_TOZERO)[1]]

    def __to_dict__(self) -> Dict:
        return {
            "name": "threshold",
            "args": {
                "threshold": self.threshold,
                "normalize": self.normalize,
                "mode": self.mode,
            },
        }


@register(Endpoint, "hough_circle")
class EndpointHoughCircle(Endpoint):
    def __init__(self, min_dist=21, p1=128, p2=64, min_radius=20, max_radius=120):
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


@register(Endpoint, "fit_ellipse")
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


@register(Endpoint, "marker_controlled_watershed")
class EndpointWatershed(Endpoint):
    def __init__(self, backend="opencv"):
        super().__init__([TypeArray, TypeArray])
        self.backend = backend

    def call(self, x):
        marker_labels = cv2.connectedComponents(x[1], connectivity=8, ltype=cv2.CV_16U)[
            1
        ]
        marker_labels[marker_labels > 255] = 0
        if self.backend == "skimage":
            labels = watershed(
                -x[0],
                markers=marker_labels,
                mask=x[0] > 0,
                watershed_line=True,
            )
            return [labels]
        elif self.backend == "opencv":
            background = x[0] <= 0
            image = cv2.merge((x[0], x[0], x[0]))
            labels = cv2.watershed(image, marker_labels.astype(np.int32))
            labels[labels <= -1] = 0
            labels[background] = 0
            return [labels]


@register(Endpoint, "local-max_watershed")
class LocalMaxWatershed(Endpoint):
    """Watershed based KartezioEndpoint, but only based on one single mask.
    Markers are computed as the local max of the distance transform of the mask

    """
    def __init__(self, markers_distance: int = 21):
        super().__init__([TypeArray])

    def call(self, x):
        distance_transform = ndimage.distance_transform_edt(x[0])

        mask = threshold_tozero(x[0], self.threshold)
        mask, markers, labels = self.wt.apply(mask, markers=None, mask=mask > 0)
        return [labels]

    def __to_dict__(self) -> Dict:
        return {
            "name": "local-max_watershed",
            "args": {
                "threshold": self.threshold,
                "markers_distance": self.wt.markers_distance,
            },
        }


def _peak_local_max(image, min_distance=21):
    peak_idx = peak_local_max(
        image,
        min_distance=min_distance,
        exclude_border=0,
    )
    peak_mask = np.zeros_like(image, dtype=np.uint8)
    labels = list(range(1, peak_idx.shape[0] + 1))
    peak_mask[tuple(peak_idx.T)] = labels
    return peak_mask


def _fast_local_max(image, min_distance=21):
    image_down = cv2.pyrDown(image)
    peak_idx = peak_local_max(
        image_down,
        min_distance=min_distance//2,
        exclude_border=0,
    )
    peak_mask = np.zeros_like(image, dtype=np.uint8)
    labels = list(range(1, peak_idx.shape[0] + 1))
    remaped_peaks = (peak_idx * 2).astype(np.int32)
    peak_mask[tuple(remaped_peaks.T)] = labels
    return peak_mask


@register(Endpoint, "raw_watershed")
class RawWatershed(Endpoint):
    def __init__(self):
        super().__init__([TypeArray])

    def call(self, x):
        marker_labels = _fast_local_max(x[0])
        labels = watershed(
            -x[0],
            markers=marker_labels,
            mask=x[0] > 0,
            watershed_line=True,
        )
        return [labels]
        

@register(Endpoint, "raw_watershed_old")
class RawLocalMaxWatershed(Endpoint):
    """Watershed based KartezioEndpoint, but only based on one single mask.
    Markers are computed as the local max of the mask

    """

    def __init__(self, threshold=1, markers_distance=21):
        super().__init__([TypeArray])
        self.wt = WatershedSkimage(markers_distance=markers_distance)
        self.threshold = threshold

    def call(self, x):
        mask = threshold_tozero(x[0], self.threshold)
        mask, markers, labels = self.wt.apply(mask, markers=None, mask=mask > 0)
        return {
            "mask_raw": x[0],
            "mask": mask,
            "markers": markers,
            "count": len(np.unique(labels)) - 1,
            "labels": labels,
        }

    def _to_json_kwargs(self) -> dict:
        return {
            "threshold": self.threshold,
            "markers_distance": self.wt.markers_distance,
        }


@register(Endpoint, "hough_circle_small")
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


@register(Endpoint, "rescale")
class EndpointRescale(Endpoint):
    def __init__(self, scale, method):
        super().__init__([TypeArray])
        self.resize = Resize(scale, method)

    def call(self, x):
        return self.resize.call([x])[0]
