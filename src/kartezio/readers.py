import ast

import cv2
import numpy as np

from kartezio.data.dataset import DataItem, DataReader
from kartezio.utils.image import imread_gray, imread_rgb, imread_tiff
from kartezio.utils.imagej import read_polygons_from_roi
from kartezio.vision.common import (
    fill_polygons_as_labels,
    gray2rgb,
    image_new,
    image_split,
)


class ImageMaskReader(DataReader):
    def _read(self, filepath, shape=None):
        if filepath == "":
            mask = image_new(shape)
            return DataItem([mask], shape, 0)
        image = imread_gray(filepath)
        _, labels = cv2.connectedComponents(image)
        return DataItem(
            [labels], image.shape[:2], len(np.unique(labels)) - 1, image
        )


class ImageLabels(DataReader):
    def _read(self, filepath, shape=None):
        image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
        labels = np.zeros_like(image, dtype=np.uint16)
        for i, current_value in enumerate(np.unique(image)):
            if current_value == 0:
                continue
            if i == 0:
                continue
            labels[image == current_value] = i
        return DataItem([labels], image.shape[:2], labels.max(), visual=image)


class ImageRGBReader(DataReader):
    def _read(self, filepath, shape=None):
        image = imread_rgb(filepath)
        return DataItem(
            image_split(image), image.shape[:2], None, visual=image
        )


class ImageGrayscaleReader(DataReader):
    def _read(self, filepath, shape=None):
        image = imread_gray(filepath)
        visual = cv2.merge((image, image, image))
        return DataItem([image], image.shape, None, visual=visual)


class RoiPolygonReader(DataReader):
    def _read(self, filepath, shape=None):
        label_mask = image_new(shape)
        if filepath == "":
            return DataItem([label_mask], shape, 0)
        polygons = read_polygons_from_roi(filepath)
        fill_polygons_as_labels(label_mask, polygons)
        return DataItem([label_mask], shape, len(polygons))


class OneHotVectorReader(DataReader):
    def _read(self, filepath, shape=None):
        label = np.array(ast.literal_eval(filepath.split("/")[-1]))
        return DataItem([label], shape, None)


class ImageChannelsReader(DataReader):
    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)
        if image.dtype == np.uint16:
            raise ValueError(f"Image must be 8bits! ({filepath})")
        shape = image.shape[-2:]
        if len(image.shape) == 2:
            channels = [image]
            preview = gray2rgb(channels[0])
        if len(image.shape) == 3:
            # channels: (c, h, w)
            channels = [channel for channel in image]
            preview = cv2.merge(
                (image_new(channels[0].shape), channels[0], channels[1])
            )
        if len(image.shape) == 4:
            # stack: (z, c, h, w)
            channels = [image[:, i] for i in range(len(image[0]))]
            preview = cv2.merge(
                (
                    channels[0].max(axis=0).astype(np.uint8),
                    channels[1].max(axis=0).astype(np.uint8),
                    image_new(channels[0][0].shape, dtype=np.uint8),
                )
            )
        return DataItem(channels, shape, None, visual=preview)
