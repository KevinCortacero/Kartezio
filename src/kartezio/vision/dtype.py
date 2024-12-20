"""Data type conversion functions for images."""


import cv2
import numpy as np


class DType:
    """Data type conversion functions for images."""
    def __init__(self, range, dtype):
        self.range = range
        self.dtype = dtype
        self.vrange = range[1] - range[0]

    def __call__(self, image):
        target_dtype = self._remap_dtype(image)
        return cv2.normalize(
            image,
            None,
            alpha=target_dtype.range[0],
            beta=target_dtype.range[1],
            norm_type=cv2.NORM_MINMAX,
            dtype=target_dtype.dtype,
        )
    
    def _remap_dtype(self, image) -> "DType":
        """Remap the data type of the image."""
        dtype = _dtype_from_image(image)
        alpha = _remap_values(dtype, self, np.min(image))
        beta = _remap_values(dtype, self, np.max(image))
        print(alpha, beta)
        return DType((alpha, beta), self.dtype)


def _remap_values(src: DType, dst: DType, value):
    return (value - src.range[0]) / src.vrange * dst.vrange + dst.range[0]


u8 = DType((0.0, 255.0), cv2.CV_8U)
u16 = DType((0.0, 65535.0), cv2.CV_16U)
uf32 = DType((0.0, 1.0), cv2.CV_32F)
f64 = DType((-1.0, 1.0), cv2.CV_64F)


def _dtype_from_image(image) -> DType:
    if image.dtype == np.uint8:
        return u8
    elif image.dtype == np.uint16:
        return u16
    elif image.dtype == np.float32:
        return uf32
    elif image.dtype == np.float64:
        return f64
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")


if __name__ == "__main__":
    image_test = np.arange(64, 192).reshape(8, 16).astype(np.uint8)
    print(image_test)
    print(image_test.dtype)
    print(uf32(image_test))
    print(uf32(image_test).dtype)
    print(u16(image_test))
    print(u16(image_test).dtype)
    print(f64(u8(u16(uf32(u8(u16(uf32(image_test))))))))



