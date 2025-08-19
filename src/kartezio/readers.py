import re

import cv2
import numpy as np
from roifile import ImagejRoi

from kartezio.data.dataset import DataItem, DataReader
from kartezio.utils.image import imread_gray, imread_rgb, imread_tiff
from kartezio.utils.imagej import read_polygons_from_roi
from kartezio.vision.common import (
    contours_as_labels_and_foreground,
    fill_polygons_as_labels,
    fill_polyhedron_as_labels,
    gray2rgb,
    image_new,
    image_split,
)


def _parse_numeric_list_from_filename(filename):
    """
    Safely parse a numeric list from a filename.

    Expected format: "prefix_[1,2,3]suffix" or just "[1,2,3]"

    Args:
        filename (str): The filename to parse

    Returns:
        list: List of numbers parsed from filename

    Raises:
        ValueError: If filename format is invalid or contains non-numeric values

    Security note: This replaces the vulnerable ast.literal_eval() usage
    with a safe regex-based parser that only accepts numeric lists.
    """
    if not filename:
        raise ValueError("Empty filename provided")

    # Extract the part that should contain the numeric list
    # Look for pattern like [1,2,3] or [1, 2, 3]
    pattern = r"\[([0-9\s,.-]+)\]"
    match = re.search(pattern, filename)

    if not match:
        raise ValueError(
            f"No numeric list pattern found in filename: {filename}"
        )

    list_content = match.group(1).strip()

    if not list_content:
        raise ValueError(f"Empty list found in filename: {filename}")

    # Split by comma and parse each number
    try:
        # Split by comma, strip whitespace, and convert to float
        numbers = []
        for item in list_content.split(","):
            item = item.strip()
            if not item:
                continue  # Skip empty items
            # Only allow numeric values (including negative and decimal)
            if not re.match(r"^-?[0-9]+\.?[0-9]*$", item):
                raise ValueError(f"Non-numeric value found: '{item}'")
            numbers.append(float(item))

        if not numbers:
            raise ValueError("No valid numbers found in list")

        return numbers

    except ValueError as e:
        raise ValueError(
            f"Failed to parse numeric values from '{list_content}': {e}"
        )
    except Exception as e:
        raise ValueError(
            f"Unexpected error parsing filename '{filename}': {e}"
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
        # Extract filename from path
        filename = filepath.split("/")[-1]

        # Safely parse the numeric list from filename
        try:
            numbers = _parse_numeric_list_from_filename(filename)
            label = np.array(numbers)
            return DataItem([label], shape, None)
        except ValueError as e:
            # Provide helpful error message for debugging
            raise ValueError(
                f"Failed to parse label from filename '{filename}': {e}"
            )


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
            channels = [channel for channel in image]
            preview = cv2.merge(
                (image_new(channels[0].shape), channels[0], channels[1])
            )
        if len(image.shape) == 4:
            channels = [image[:, i] for i in range(len(image[0]))]
            preview = cv2.merge(
                (
                    channels[0].max(axis=0).astype(np.uint8),
                    channels[1].max(axis=0).astype(np.uint8),
                    image_new(channels[0][0].shape, dtype=np.uint8),
                )
            )
        return DataItem(channels, shape, None, visual=preview)


# New feature - to be tested


class RoiPolyhedronReader(DataReader):
    """
    3D datareader , label reader from ROI.
    load ROI mask in 3D , roi_name must be  numberLabels_Znumber  ex 1_Z2  label 1 in z slice 2
    generate label polygon on each z slice
    """

    def _read(self, filepath, shape=None):
        label_mask = image_new(shape)
        if filepath == "":
            return DataItem([label_mask], shape, 0)
        rois = ImagejRoi.fromfile(filepath)
        if isinstance(rois, ImagejRoi):
            return [rois.coordinates()]
        contours = [roi.coordinates() for roi in rois]
        labels = [
            int(roi.name.split("_")[0]) for roi in rois
        ]  # name in regex #label_Z#slice
        z_slice = [int(roi.name.split("Z")[-1]) - 1 for roi in rois]
        label_mask = image_new(shape)
        label_mask = fill_polyhedron_as_labels(
            label_mask, labels, z_slice, contours
        )
        return DataItem([label_mask], shape, len(contours))


class TiffImageChannelsMask3dReader(DataReader):
    """
    3D datareader , data reader from ROI.
    load ROI mask in 3D , roi_name must be  numberLabels_Znumber  ex 1_Z2  label 1 in z slice 2
    generate label polygon on each z slice
    """

    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)
        if image.dtype == np.uint16:
            raise ValueError(f"Image must be 8bits! ({filepath})")
        shape = (image.shape[0],) + image.shape[-2:]
        if len(image.shape) == 2:
            channels = [image]
            previews = gray2rgb(channels[0])
        if len(image.shape) == 3:
            channels = [channel for channel in image]
            previews = cv2.merge(
                (image_new(channels[0].shape), channels[0], channels[1])
            )
        if len(image.shape) == 4:
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
            previews = np.asarray(previews).reshape(shape + (3,))
        return DataItem(channels, shape, None, visual=previews)


class TiffImageGray3dReader(DataReader):
    """
    3D datareader , image reader tiff mono channel shape (z,h,w)
    """

    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)
        if image.dtype == np.uint16:
            raise ValueError(f"Image must be 8bits! ({filepath})")
        shape = (image.shape[0],) + image.shape[-2:]
        if len(image.shape) == 3:
            previews = []
            for z in range(image.shape[0]):
                preview = (image[z].astype(np.uint8),)

                previews.append(preview)
            previews = np.asarray(previews).reshape(shape)
        else:
            raise ValueError(f"Image must be shape (z,h,w) ({filepath})")
        return DataItem([image], shape, None, visual=previews)


class TiffImageLabel3dReader(DataReader):
    """
    3D datareader , label reader from tiff images.
    """

    def _read(self, filepath, shape=None):
        image = imread_tiff(filepath)

        # Get unique values
        unique_values = np.unique(image)
        # Generate the expected range
        expected_values = np.arange(
            unique_values.min(), unique_values.max() + 1
        )
        # Check if unique values match the expected range, to avoid problem in fitness calcul, labels need to be
        # in continue series of int [ 0,1,2,3..n]
        is_continuous = np.array_equal(unique_values, expected_values)
        if not is_continuous:
            for i, current_value in enumerate(unique_values):
                image[image == current_value] = i
        return DataItem([image], shape, image.max(), visual=image)


class RoiForegroundOutlineReader(DataReader):
    """
    2D datareader , label reader ,  item labelling 1 , outlines labelling 2 , foreground labelling 0
    """

    def _read(self, filepath, shape=None):
        label_mask = image_new(shape)
        if filepath == "":
            return DataItem([label_mask], shape, 0)
        contours = read_polygons_from_roi(filepath)
        label_mask = contours_as_labels_and_foreground(label_mask, contours)
        return DataItem([label_mask], shape, len(contours))
