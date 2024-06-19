import cv2
import numpy as np

from kartezio.core.components.base import register
from kartezio.core.components.preprocessing import Preprocessing
from kartezio.vision.common import (
    bgr2hed,
    rgb2hsv,
    hsv2rgb,
    image_split,
    rgb2bgr,
    rgb2hed,
)


@register(Preprocessing, "pyr_shift")
class PyrMeanShift(Preprocessing):
    def __init__(self, sp=2, sr=16):
        super().__init__()
        self.sp = sp
        self.sr = sr

    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            filtered = cv2.pyrMeanShiftFiltering(original_image, sp=self.sp, sr=self.sr)
            cv2.imwrite(f"pyr_{i}.png", filtered)
            new_x.append(image_split(filtered))
        return new_x


@register(Preprocessing, "pyr_scale")
class PyrScale(Preprocessing):
    def __init__(self, level, scale: str):
        super().__init__()
        self.level = level
        self.scale = scale

    def preprocess(self, x, args=None):
        if self.level == 0:
            return x
        new_x = []
        for i in range(len(x)):
            new_xi = []
            for xij in x[i]:
                for l in range(self.level):
                    if self.scale == "down":
                        xij = cv2.pyrDown(xij.astype(np.uint8))
                    elif self.scale == "up":
                        xij = cv2.pyrUp(xij.astype(np.uint8))
                new_xi.append(xij)
            new_x.append(new_xi)
        return new_x



@register(Preprocessing, "to_hsv")
class TransformToHSV(Preprocessing):
    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            transformed = rgb2hsv(original_image)
            new_x.append(image_split(transformed))
        return new_x
    

@register(Preprocessing, "clahe_value")
class ClaheValue(Preprocessing):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            transformed = rgb2hsv(original_image)
            transformed[:, :, 2] = self.clahe.apply(transformed[:, :, 2])
            new_x.append(image_split(hsv2rgb(transformed)))
        return new_x


@register(Preprocessing, "to_hed")
class TransformToHED(Preprocessing):
    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            transformed = rgb2hed(original_image)
            new_x.append(image_split(transformed))
        return new_x
    

@register(Preprocessing, "to_gray")
class ToGray(Preprocessing):
    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            transformed = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            new_x.append([transformed])
        return new_x
    
@register(Preprocessing, "append_hed")
class AppendHED(Preprocessing):
    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i][:3])
            transformed = rgb2hed(original_image)
            new_x.append(x[i] + image_split(transformed))
        return new_x


@register(Preprocessing, "append_gray")
class AppendGray(Preprocessing):
    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i][:3])
            transformed = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            new_x.append(x[i] + [transformed])
        return new_x
    

@register(Preprocessing, "append_hsv")
class AppendHSV(Preprocessing):
    def preprocess(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i][:3])
            transformed = rgb2hsv(original_image)
            new_x.append(x[i] + image_split(transformed))
        return new_x


@register(Preprocessing, "select_channels")
class SelectChannels(Preprocessing):
    def __init__(self, channels, scalars=None):
        super().__init__()
        self.channels = channels
        self.scalars = scalars

    def preprocess(self, x):
        new_x = []
        for i in range(len(x)):
            one_item = [x[i][channel] for channel in self.channels]
            if self.scalars:
                # labels = [np.zeros_like(channel, dtype=np.int16) for channel in self.channels]
                # new_x.append([one_item, self.scalars, labels])
                new_x.append([one_item, self.scalars])
            else:
                new_x.append(one_item)
        return new_x


class Format3D(Preprocessing):
    def __init__(self, channels=None, z_range=None):
        super().__init__("Format to 3D", "F3D")
        self.channels = channels
        self.z_range = z_range

    def preprocess(self, x, args=None):
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
