from typing import List

import numpy as np

from kartezio.core.components import Reducer, register
from kartezio.vision.common import threshold_tozero


@register(Reducer)
class BasicReducton(Reducer):
    def _to_json_kwargs(self) -> dict:
        return {
            "mode": self.mode,
            "threshold": self.threshold,
        }

    def __init__(self, mode="mean", threshold=4):
        super().__init__()
        self.mode = mode
        self.threshold = threshold

    def reduce(self, x: List):
        if self.mode == "mean":
            return np.mean(np.array(x), axis=0).astype(np.uint8)
        elif self.mode == "sum":
            return np.sum(np.array(x), axis=0).astype(np.uint8)
        elif self.mode == "min":
            return np.min(np.array(x), axis=0).astype(np.uint8)
        elif self.mode == "max":
            return np.max(np.array(x), axis=0).astype(np.uint8)
        elif self.mode == "median":
            return np.median(np.array(x), axis=0).astype(np.uint8)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def post_reduction(self, x, index):
        return threshold_tozero(x, self.threshold)
