from typing import List

import cv2
import numpy as np
from numena.image.morphology import morph_erode
from numena.image.threshold import threshold_tozero

from kartezio.model.components import KartezioStacker
from kartezio.model.registry import registry


def register_stackers():
    print(f"[Kartezio - INFO] -  {len(registry.stackers.list())} stackers registered.")


@registry.stackers.add("MEAN")
class StackerMean(KartezioStacker):
    def _to_json_kwargs(self) -> dict:
        return {}

    def __init__(self, name="mean_stacker", symbol="MEAN", arity=1, threshold=4):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y: List):
        return np.mean(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        return threshold_tozero(yi, self.threshold)


@registry.stackers.add("SUM")
class StackerSum(KartezioStacker):
    def _to_json_kwargs(self) -> dict:
        return {}

    def __init__(self, name="Sum KartezioStacker", symbol="SUM", arity=1, threshold=4):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y: List):
        stack_array = np.array(Y).astype(np.float32)
        stack_array /= 255.
        stack_sum = np.sum(stack_array, axis=0)
        return stack_sum.astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        if index == 0:
            return cv2.GaussianBlur(yi, (7, 7), 1)
        return yi


@registry.stackers.add("MIN")
class StackerMin(KartezioStacker):
    def _to_json_kwargs(self) -> dict:
        return {}

    def __init__(self, name="min_stacker", symbol="MIN", arity=1, threshold=4):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y: List):
        return np.min(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        return threshold_tozero(yi, self.threshold)


@registry.stackers.add("MAX")
class StackerMax(KartezioStacker):
    def _to_json_kwargs(self) -> dict:
        return {}

    def __init__(self, name="max_stacker", symbol="MAX", arity=1, threshold=1):
        super().__init__(name, symbol, arity)
        self.threshold = threshold

    def stack(self, Y: List):
        return np.max(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        if index == 0:
            return cv2.GaussianBlur(yi, (7, 7), 1)
        return yi


@registry.stackers.add("MEANW")
class MeanKartezioStackerForWatershed(KartezioStacker):
    def _to_json_kwargs(self) -> dict:
        return {"half_kernel_size": self.half_kernel_size, "threshold": self.threshold}

    def __init__(self, half_kernel_size=1, threshold=4):
        super().__init__(name="mean_stacker_watershed", symbol="MEANW", arity=2)
        self.half_kernel_size = half_kernel_size
        self.threshold = threshold

    def stack(self, Y: List):
        return np.mean(np.array(Y), axis=0).astype(np.uint8)

    def post_stack(self, x, index):
        yi = x.copy()
        if index == 1:
            # supposed markers
            yi = morph_erode(yi, half_kernel_size=self.half_kernel_size)
        return threshold_tozero(yi, self.threshold)
