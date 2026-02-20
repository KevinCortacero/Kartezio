import numpy as np

from kartezio.core.components import Reducer, register
from kartezio.types import DataSequence


@register(Reducer)
class BasicReduction(Reducer):
    def _to_json_kwargs(self) -> dict:
        return {"mode": self.mode}

    def __init__(self, mode="mean", threshold=4):
        super().__init__()
        self.mode = mode

    def reduce(self, x: DataSequence):
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
            raise ValueError(f"Unknown reduction mode {self.mode}")
