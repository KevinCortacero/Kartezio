from typing import List

import numpy as np

from kartezio.core.components import Library, Primitive, register
from kartezio.types import TypeArray, TypeFourier, TypeScalar


@register(Primitive)
class FFT(Primitive):
    def __init__(self):
        super().__init__([TypeArray], TypeFourier, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        # Convert to float and normalize to range [0, 1]
        image = x[0].astype(np.float32) / 255.0
        # Perform FFT and shift the zero frequency component to the center
        f_transform = np.fft.fftshift(np.fft.fft2(image))
        return f_transform


@register(Primitive)
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


@register(Primitive)
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


@register(Primitive)
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


@register(Primitive)
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


@register(Primitive)
class MultiplyFourier(Primitive):
    def __init__(self):
        super().__init__([TypeFourier, TypeFourier], TypeFourier, 0)

    def call(self, x: List[np.ndarray], args: List[int]):
        return x[0] * x[1]


library_fourier = Library(TypeFourier)
library_fourier.add_primitive(FFT())
library_fourier.add_primitive(LowPass())
library_fourier.add_primitive(HighPass())
library_fourier.add_primitive(BandPass())
library_fourier.add_primitive(PhaseCongruency())
library_fourier.add_primitive(MultiplyFourier())
print(library_fourier.display())
