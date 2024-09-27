import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.transform import hough_ellipse

from kartezio.core.components.base import register
from kartezio.core.components.endpoint import Endpoint
from kartezio.core.types import TypeArray, TypeLabels
from kartezio.libraries.array import Sobel
from kartezio.vision.common import (
    WatershedSkimage,
    contours_find,
    image_new,
    threshold_tozero,
)


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
            return [
                cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)[1]
            ]
        return [
            cv2.threshold(image, self.threshold, 255, cv2.THRESH_TOZERO)[1]
        ]


@register(Endpoint, "hough_circle")
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


@register(Endpoint, "fit_ellipse")
class EndpointEllipse(Endpoint):
    def __init__(self, min_axis=10, max_axis=30, backend="opencv"):
        super().__init__([TypeArray])
        self.min_axis = min_axis
        self.max_axis = max_axis
        self.backend = backend
        self.edge_detector = Sobel()

    def call(self, x, args=None):
        mask = x[0]
        n = 0
        new_labels = image_new(mask.shape)
        labels = []
        edges = self.edge_detector.call([mask], [3, 3])
        if self.backend == "opencv":
            cnts = contours_find(edges, exclude_holes=True)
            for cnt in cnts:
                if len(cnt) >= 5:
                    center, (MA, ma), angle = cv2.fitEllipse(cnt)
                    if (
                        self.min_axis <= MA <= self.max_axis
                        and self.min_axis <= ma <= self.max_axis
                    ):
                        cv2.ellipse(
                            new_labels,
                            (center, (MA, ma), angle),
                            n + 1,
                            thickness=-1,
                        )
                        labels.append((center, (MA, ma), angle))
                        n += 1

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

        return [new_labels]


@register(Endpoint, "marker_controlled_watershed")
class EndpointWatershed(Endpoint):
    def __init__(self, backend="opencv"):
        super().__init__([TypeArray, TypeArray])
        self.backend = backend

    def call(self, x):
        marker_labels = cv2.connectedComponents(
            x[1], connectivity=8, ltype=cv2.CV_16U
        )[1]
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


@register(Endpoint, "raw_watershed")
class EndpointRawWatershed(Endpoint):
    def __init__(self):
        super().__init__([TypeArray, TypeLabels])

    def call(self, x):
        background = x[0] == 0
        image = cv2.merge((x[0], x[0], x[0]))
        labels = cv2.watershed(image, x[1])
        labels[labels == -1] = 0
        labels[background] = 0
        return [labels]


@register(Endpoint, "local-max_watershed")
class LocalMaxWatershed(Endpoint):
    """Watershed based KartezioEndpoint, but only based on one single mask.
    Markers are computed as the local max of the distance transform of the mask

    """

    def __init__(self, threshold: int = 1, markers_distance: int = 21):
        super().__init__([TypeArray])
        self.wt = WatershedSkimage(
            use_dt=True, markers_distance=markers_distance
        )
        self.threshold = threshold

    def call(self, x):
        mask = threshold_tozero(x[0], self.threshold)
        mask, markers, labels = self.wt.apply(
            mask, markers=None, mask=mask > 0
        )
        return [labels]
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
        mask, markers, labels = self.wt.apply(
            mask, markers=None, mask=mask > 0
        )
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
