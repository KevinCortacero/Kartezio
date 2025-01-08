import ast

import cv2
import numpy as np

from kartezio.data.dataset import DataItem, DataReader
from kartezio.utils.image import imread_gray, imread_rgb, imread_tiff
from kartezio.utils.imagej import read_polygons_from_roi
from kartezio.vision.common import (
    fill_polygons_as_labels,
    fill_polyhedron_as_labels,
    gray2rgb,
    image_new,
    image_split,
)
from roifile import ImagejRoi

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
        for i, current_value in enumerate(np.unique(image)):
            image[image == current_value] = i
        return DataItem([image], image.shape[:2], image.max(), visual=image)


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

### nouveauté a tester

class RoiPolyhedronReader(DataReader):
    def _read(self, filepath, shape=None):
        label_mask = image_new(shape)
        if filepath == "":
            return DataItem([label_mask], shape, 0)
        rois = ImagejRoi.fromfile(filepath)
        if type(rois) == ImagejRoi:
            return [rois.coordinates()]
        contours = [roi.coordinates() for roi in rois]
        labels = [int(roi.name.split('_')[0]) for roi in rois]  # name in regex #label_Z#slice
        z_slice = [roi.z_position - 1 for roi in rois]
        label_mask = image_new(shape)
        label_mask = fill_polyhedron_as_labels(label_mask,labels,z_slice, contours)
        return DataItem([label_mask], shape, len(contours))



class ImageChannelsMask3dReader(DataReader):
    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)
        if image.dtype == np.uint16:
            raise ValueError(f"Image must be 8bits! ({filepath})")
        shape = (image.shape[0],) + image.shape[-2:]
        if len(image.shape) == 2:
            channels = [image]
            previews = gray2rgb(channels[0])
        if len(image.shape) == 3:
            # channels: (c, h, w)
            channels = [channel for channel in image]
            previews = cv2.merge(
                (image_new(channels[0].shape), channels[0], channels[1])
            )
        if len(image.shape) == 4:
            # stack: (z, c, h, w)
            channels = [image[:, i] for i in range(len(image[0]))]
            previews = []
            for z in range(image.shape[0]):

                preview = cv2.merge(
                    (
                        channels[0][z].astype(np.uint8),
                        channels[0][z].astype(np.uint8),
                        image_new(channels[0][0].shape, dtype=np.uint8),
                    )
                )
                previews.append(preview)
            previews=np.asarray(previews).reshape(shape+(3,))
            #cv2.imwrite("rgb_image.png", preview)
        return DataItem(channels, shape, None, visual=previews)



class ImageGray3dReader(DataReader):
    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)
        if image.dtype == np.uint16:
            raise ValueError(f"Image must be 8bits! ({filepath})")
        shape = (image.shape[0],) + image.shape[-2:]
        if len(image.shape) == 3:
            # (z, h, w)
            previews = []
            for z in range(image.shape[0]):
                preview = image[z].astype(np.uint8),

                previews.append(preview)
            previews = np.asarray(previews).reshape(shape)
        else :
            raise ValueError(f"Image must be shape (z,h,w) ({filepath})")
        return DataItem([image], shape, None, visual=previews)



class ImageLabel3dReader(DataReader):
    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)
        for i, current_value in enumerate(np.unique(image)):
            image[image == current_value] = i
        return DataItem([image], shape, image.max(), visual=image)



class ImageGray3dCutReader(DataReader):
    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)
        if image.dtype == np.uint16:
            raise ValueError(f"Image must be 8bits! ({filepath})")
        shape = (image.shape[0],) + image.shape[-2:]
        z, c, x, y = image.shape
        # Ensure that x and y dimensions are divisible by 2
        assert x % 2 == 0 and y % 2 == 0, \
            "x and y dimensions must be divisible by 2"
        # Calculate half dimensions
        if len(image.shape) != 3:
            raise ValueError(f"Image must be shape (z,h,w) ({filepath})")
        half_x, half_y = x // 2, y // 2
        # (z, h, w)
        previews = []
        for z in range(image.shape[0]):
            preview = image[z].astype(np.uint8),

            previews.append(preview)
        previews = np.asarray(previews).reshape(shape)

        return DataItem([image], shape, None, visual=previews)