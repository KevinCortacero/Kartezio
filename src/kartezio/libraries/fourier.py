from typing import List

import numpy as np

from kartezio.components.base import register
from kartezio.components.library import Library, Primitive
from kartezio.core.types import TypeArray, TypeFourier, TypeScalar


@register(Library, "opencv_fourier")
class LibraryFourier(Library):
    def __init__(self):
        super().__init__(TypeFourier)


@register(Primitive, "fft")
class FFT(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeFourier, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        # Convert to float and normalize to range [0, 1]
        image = x[0].astype(np.float32) / 255.0
        # Perform FFT and shift the zero frequency component to the center
        f_transform = np.fft.fftshift(np.fft.fft2(image))
        return f_transform


@register(Primitive, "low_pass")
class LowPass(Primitive):
    def __init__(self):
        super().__init__([TypeFourier, TypeScalar], TypeFourier, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        rows, cols = x[0].shape
        crow, ccol = rows // 2, cols // 2
        filter_mask = np.zeros((rows, cols), np.float32)
        x, y = np.ogrid[:rows, :cols]
        center = (crow, ccol)
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        filter_mask = np.exp(-(distance**2) / (2 * (x[1] / 2.0) ** 2))
        filtered = x[0] * filter_mask
        return filtered


@register(Primitive, "high_pass")
class HighPass(Primitive):
    def __init__(self):
        super().__init__([TypeFourier, TypeScalar], TypeFourier, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        rows, cols = x[0].shape
        crow, ccol = rows // 2, cols // 2
        filter_mask = np.zeros((rows, cols), np.float32)
        x, y = np.ogrid[:rows, :cols]
        center = (crow, ccol)
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        filter_mask = 1 - np.exp(-(distance**2) / (2 * (x[1] / 2.0) ** 2))
        filtered = x[0] * filter_mask
        return filtered


@register(Primitive, "band_pass")
class BandPass(Primitive):
    def __init__(self):
        super().__init__([TypeFourier, TypeScalar], TypeFourier, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        rows, cols = x[0].shape
        crow, ccol = rows // 2, cols // 2
        filter_mask = np.zeros((rows, cols), np.float32)
        x, y = np.ogrid[:rows, :cols]
        center = (crow, ccol)
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        filter_mask = np.exp(
            -(distance**2) / (2 * (x[1] * 2 / 2.0) ** 2)
        ) - np.exp(-(distance**2) / (2 * (x[1] / 2.0) ** 2))
        filtered = x[0] * filter_mask
        return filtered


@register(Primitive, "phase_congruency")
class PhaseCongruency(Primitive):
    def __init__(self):
        super().__init__([TypeFourier], TypeFourier, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        # Compute magnitude and phase
        magnitude = np.abs(x[0])
        phase = np.angle(x[0])

        # Compute phase congruency
        # Here we use a simple phase congruency calculation for illustration
        phase_congruency = magnitude * np.cos(phase)
        return phase_congruency


@register(Primitive, "multiply_fourier")
class MultiplyFourrier(Primitive):
    def __init__(self):
        super().__init__([TypeFourier, TypeFourier], TypeFourier, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return x[0] * x[1]


library_fourier = LibraryFourier()
library_fourier.add_by_name("fft")
library_fourier.add_by_name("low_pass")
library_fourier.add_by_name("high_pass")
library_fourier.add_by_name("band_pass")
library_fourier.add_by_name("phase_congruency")
library_fourier.add_by_name("multiply_fourier")
# library_fourrier.add_signatures("bitwise_and", [TypeFourier, TypeFourier], TypeFourier, 0)
