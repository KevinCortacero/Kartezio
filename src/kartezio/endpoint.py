from typing import List

import cv2
import numpy as np
from numena.enums import IMAGE_UINT8_COLOR_1C
from numena.image.basics import image_new
from numena.image.contour import contours_find
from numena.image.morphology import WatershedSkimage
from numena.image.threshold import threshold_tozero

from kartezio.core.components.base import Components, register
from kartezio.core.components.endpoint import Endpoint
from kartezio.core.types import TypeArray


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


@register(Endpoint, "threshold")
class EndpointThreshold(Endpoint):
    def call(self, x):
        if self.mode == "binary":
            return [cv2.threshold(x[0], self.threshold, 255, cv2.THRESH_BINARY)[1]]
        return [cv2.threshold(x[0], self.threshold, 255, cv2.THRESH_TOZERO)[1]]

    def __init__(self, threshold, mode="binary"):
        super().__init__([TypeArray])
        self.threshold = threshold
        self.mode = mode


endpoint_labels = ToLabels()

endpoint_threshold = EndpointThreshold(128)


@register(Endpoint, "hough_circle")
class EndpointHoughCircle(Endpoint):
    def __init__(self, min_dist=21, p1=128, p2=64, min_radius=20, max_radius=120):
        super().__init__([TypeArray])
        self.min_dist = min_dist
        self.p1 = p1
        self.p2 = p2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def _to_json_kwargs(self) -> dict:
        return {
            "min_dist": self.min_dist,
            "p1": self.p1,
            "p2": self.p2,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
        }

    def call(self, x):
        mask = x[0]
        n = 0
        new_mask = image_new(mask.infos)
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

        return {
            "mask_raw": mask,
            "labels": new_mask,
            "count": n,
        }


@register(Endpoint, "fit_ellipse")
class EndpointEllipse(Endpoint):
    def _to_json_kwargs(self) -> dict:
        return {
            "min_axis": self.min_axis,
            "max_axis": self.max_axis,
        }

    def __init__(self, min_axis=10, max_axis=30):
        super().__init__([TypeArray])
        self.min_axis = min_axis
        self.max_axis = max_axis

    def call(self, x, args=None):
        mask = x[0]
        n = 0
        new_labels = image_new(mask.infos)
        labels = []

        cnts = contours_find(x[0], exclude_holes=True)
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
        new_mask = new_labels.copy().astype(np.uint8)
        new_mask[new_mask > 0] = IMAGE_UINT8_COLOR_1C
        return {
            "mask_raw": mask,
            "mask": new_mask,
            "labels": new_labels,
            "count": n,
        }


def marker_controlled_watershed(x: List):
    mask_raw = x[0]
    markers_raw = x[1]
    wt = WatershedSkimage(use_dt=False, markers_distance=21, markers_area=None)
    mask, markers, labels = wt.apply(mask_raw, markers=markers_raw, mask=mask_raw > 0)
    return {
        "mask_raw": mask_raw,
        "markers_raw": markers_raw,
        "mask": mask,
        "markers": markers,
        "count": labels.max(),
        "labels": labels,
    }


e_mcw = Endpoint(marker_controlled_watershed, [TypeArray, TypeArray])


@registry.endpoints.add("WSHD")
class EndpointWatershed(Endpoint):
    def __init__(self, use_dt=False, markers_distance=21, markers_area=None):
        super().__init__("Marker-Based Watershed", "WSHD", 2, [])
        self.wt = WatershedSkimage(
            use_dt=use_dt, markers_distance=markers_distance, markers_area=markers_area
        )

    def call(self, x, args=None):
        mask = x[0]
        markers = x[1]
        mask, markers, labels = self.wt.apply(mask, markers=markers, mask=mask > 0)
        return {
            "mask_raw": x[0],
            "markers_raw": x[1],
            "mask": mask,
            "markers": markers,
            "count": len(np.unique(labels)) - 1,
            "labels": labels,
        }

    def _to_json_kwargs(self) -> dict:
        return {
            "use_dt": self.wt.use_dt,
            "markers_distance": self.wt.markers_distance,
            "markers_area": self.wt.markers_area,
        }


@registry.endpoints.add("LMW")
class LocalMaxWatershed(Endpoint):
    """Watershed based KartezioEndpoint, but only based on one single mask.
    Markers are computed as the local max of the distance transform of the mask

    """

    def __init__(self, threshold=1, markers_distance=21):
        super().__init__("Local-Max Watershed", "LMW", 1, [])
        self.wt = WatershedSkimage(use_dt=True, markers_distance=markers_distance)
        self.threshold = threshold

    def call(self, x, args=None):
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


@registry.endpoints.add("RLMW")
class RawLocalMaxWatershed(Endpoint):
    """Watershed based KartezioEndpoint, but only based on one single mask.
    Markers are computed as the local max of the mask

    """

    def __init__(self, threshold=1, markers_distance=21):
        super().__init__("Raw Local-Max Watershed", "RLMW", 1, [])
        self.wt = WatershedSkimage(markers_distance=markers_distance)
        self.threshold = threshold

    def call(self, x, args=None):
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
