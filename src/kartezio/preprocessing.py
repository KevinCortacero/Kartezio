from typing import Dict, List

import cv2
import numpy as np

from kartezio.core.components import Preprocessing, register
from kartezio.vision.common import image_split


@register(Preprocessing)
class PyrMeanShift(Preprocessing):
    def __init__(self, sp=2, sr=16):
        super().__init__()
        self.sp = sp
        self.sr = sr

    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i][:3])
            filtered = cv2.pyrMeanShiftFiltering(
                original_image, sp=self.sp, sr=self.sr
            )
            new_x.append(image_split(filtered))
        return new_x


@register(Preprocessing)
class PyrScale(Preprocessing):
    def __init__(self, level, scale: str, preserve_values=False):
        super().__init__()
        self.level = level
        self.scale = scale
        self.preserve_values = preserve_values

    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            new_xi = []
            for xij in x[i]:
                dsize = (xij.shape[1], xij.shape[0])
                if dsize[0] % 2 != 0:
                    dsize = (dsize[0] + 1, dsize[1])
                if dsize[1] % 2 != 0:
                    dsize = (dsize[0], dsize[1] + 1)
                if self.level == 0:
                    xij = cv2.resize(
                        xij, dsize, interpolation=cv2.INTER_NEAREST
                    )
                else:
                    for _ in range(self.level):
                        if self.preserve_values:
                            if self.scale == "down":
                                dsize = (dsize[0] // 2, dsize[1] // 2)
                                xij = cv2.resize(
                                    xij, dsize, interpolation=cv2.INTER_NEAREST
                                )
                            elif self.scale == "up":
                                dsize = (dsize[0] * 2, dsize[1] * 2)
                                xij = cv2.resize(
                                    xij, dsize, interpolation=cv2.INTER_NEAREST
                                )
                        else:
                            if self.scale == "down":
                                xij = cv2.pyrDown(xij.astype(np.uint8))
                            elif self.scale == "up":
                                xij = cv2.pyrUp(xij.astype(np.uint8))
                new_xi.append(xij)
            new_x.append(new_xi)
        return new_x


@register(Preprocessing)
class ToColorSpace(Preprocessing):
    def __init__(self, color_space):
        super().__init__()
        self.color_space = color_space

    def preprocess(self, x: List):
        if self.color_space == "rgb":
            return x
        new_x = []
        for xi in x:
            # assuming that the image (3 first elements) is in RGB
            rgb_image = cv2.merge(xi[:3])
            transformed = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            transformed = list(cv2.split(transformed))
            # append existing channels and transformed channels
            new_x.append(transformed)
        return new_x

    def __to_dict__(self) -> Dict:
        return {
            "name": "to_color_space",
            "args": {"color_space": self.color_space},
        }


@register(Preprocessing)
class AddColorSpace(Preprocessing):
    def __init__(self, color_space):
        super().__init__()
        self.color_space = color_space

    def preprocess(self, x):
        new_x = []
        for i in range(len(x)):
            # assuming that the image (3 first elements) is in RGB
            original_image = cv2.merge(x[i][:3])
            if self.color_space == "hed":
                transformed = image_split(rgb2hed(original_image))
            elif self.color_space == "hsv":
                transformed = image_split(rgb2hsv(original_image))
            elif self.color_space == "bgr":
                transformed = image_split(rgb2bgr(original_image))
            elif self.color_space == "lab":
                transformed = image_split(
                    cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB)
                )
            elif self.color_space == "gray":
                transformed = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            new_x.append(list(x[i]) + transformed)
        return new_x

    def __to_dict__(self) -> Dict:
        return {
            "name": "add_color_space",
            "args": {"color_space": self.color_space},
        }


@register(Preprocessing)
class Resize(Preprocessing):
    def __init__(self, scale, method):
        super().__init__()
        self.scale = scale
        self.method = method

    def preprocess(self, x):
        new_x = []
        for i in range(len(x)):
            new_xi = []
            for xij in x[i]:
                if self.scale == "down":
                    if self.method == "pyr":
                        new_xi.append(cv2.pyrDown(xij))
                    if self.method == "nearest":
                        new_xi.append(
                            cv2.resize(
                                xij,
                                (0, 0),
                                fx=0.5,
                                fy=0.5,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )
                elif self.scale == "up":
                    if self.method == "pyr":
                        new_xi.append(cv2.pyrUp(xij))
                    if self.method == "nearest":
                        new_xi.append(
                            cv2.resize(
                                xij,
                                (0, 0),
                                fx=2,
                                fy=2,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )
            new_x.append(new_xi)
        return new_x


@register(Preprocessing)
class ApplyClahe(Preprocessing):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def preprocess(self, x):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            transformed = transformed = cv2.cvtColor(
                original_image, cv2.COLOR_RGB2GRAY
            )
            transformed = self.clahe.apply(transformed)
            new_x.append(
                image_split(cv2.cvtColor(transformed, cv2.COLOR_GRAY2RGB))
            )
        return new_x


@register(Preprocessing)
class SelectChannels(Preprocessing):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def preprocess(self, x):
        new_x = []
        for i in range(len(x)):
            one_item = [x[i][channel] for channel in self.channels]
            new_x.append(one_item)
        return new_x

    def __to_dict__(self) -> Dict:
        return {
            "name": "select_channels",
            "args": {"channels": self.channels},
        }


@register(Preprocessing)
class Format3D(Preprocessing):
    def __init__(self, channels=None, z_range=None):
        super().__init__()
        self.channels = channels
        self.z_range = z_range

    def preprocess(self, x):
        new_x = []
        for i in range(len(x)):
            one_item = []
            if self.channels:
                if self.z_range:
                    for z in self.z_range:
                        one_item.append(
                            [x[i][channel][z] for channel in self.channels]
                        )
                else:
                    for z in range(len(x[i][0])):
                        one_item.append(
                            [x[i][channel][z] for channel in self.channels]
                        )
            else:
                if self.z_range:
                    for z in self.z_range:
                        one_item.append([x[i][:][z]])
                else:
                    for z in range(len(x[i])):
                        one_item.append([x[i][:][z]])
            new_x.append(one_item)
        return new_x

    def _to_json_kwargs(self) -> dict:
        pass

@register(Preprocessing)
class Format3DNoChannel(Preprocessing):
    """
    Preprocessing for tiff images shape ( z,h,w)
    """
    def __init__(self, z_range=None,threshold=1,axis="z"):
        super().__init__()
        self.z_range = z_range
        self.threshold = threshold
        self.axis = axis

    def preprocess(self, x):
        new_x = []
        for i in range(len(x)):
            one_item = []
            if self.axis =="z":
                for z in range(len(x[i][0])):
                    one_item.append([
                        np.where(
                            x[i][0][z,:,:] > self.threshold,
                            x[i][0][z,:,:],
                            0
                        )])
            if self.axis == "h":
                for h in range(np.shape(x[i][0])[1]):
                    one_item.append([
                        np.where(
                            x[i][0][:,h,:] > self.threshold,
                            x[i][0][:,h,:],
                            0
                        )])
            if self.axis == "w":
                for w in range(np.shape(x[i][0])[2]):
                    one_item.append([
                        np.where(
                            x[i][0][:, :, w] > self.threshold,
                            x[i][0][:, :, w],
                            0
                        )])
            new_x.append(one_item)
        return new_x


    def __to_dict__(self) -> Dict:
        return {
            "args": {"z_range": self.z_range,"threshold":self.threshold,"axis":self.axis}
        }

# @register(Preprocessing)
# class Format3DNoChannel(Preprocessing):
#     """
#     Preprocessing for tiff images shape ( z,h,w)
#     """
#     def __init__(self, z_range=None,threshold=1):
#         super().__init__()
#         self.z_range = z_range
#         self.threshold = threshold
#
#     def preprocess(self, x):
#         new_x = []
#         for i in range(len(x)):
#             one_item = []
#             if self.z_range:
#                 for z in self.z_range:
#                     one_item.append([
#                         np.where(
#                             x[i][:][z] > self.threshold,
#                             x[i][:][z],
#                             0
#                         ) if self.threshold is not None else x[i][z]
#                     ])
#             else:
#                 for z in range(len(x[i][0])):
#                     one_item.append([
#                         np.where(
#                             x[i][0][z,:,:] > self.threshold,
#                             x[i][0][z,:,:],
#                             0
#                         ) if self.threshold is not None else x[i][z]])
#             new_x.append(one_item)
#         return new_x
#
#     def __to_dict__(self) -> Dict:
#         return {
#             "args": {"z_range": self.z_range,"threshold":self.threshold}
#         }
