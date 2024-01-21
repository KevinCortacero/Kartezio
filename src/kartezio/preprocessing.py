import cv2
import numpy as np
from numena.image.basics import image_split
from numena.image.color import bgr2hed, bgr2hsv, rgb2bgr, rgb2hed

from kartezio.core.components.base import register
from kartezio.core.components.preprocessing import Preprocessing


class TransformToHSV(Preprocessing):
    def __init__(self, source_color="bgr"):
        super().__init__("Transform to HSV", "HSV")
        self.source_color = source_color

    def call(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            if self.source_color == "bgr":
                transformed = bgr2hsv(original_image)
            elif self.source_color == "rgb":
                transformed = bgr2hsv(rgb2bgr(original_image))
            new_x.append(image_split(transformed))
        return new_x

    def _to_json_kwargs(self) -> dict:
        pass


class TransformToHED(Preprocessing):
    def __init__(self, source_color="bgr"):
        super().__init__("Transform to HED", "HED")
        self.source_color = source_color

    def call(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            original_image = cv2.merge(x[i])
            if self.source_color == "bgr":
                transformed = bgr2hed(original_image)
            elif self.source_color == "rgb":
                transformed = rgb2hed(original_image)
            new_x.append(image_split(transformed))
        return new_x

    def _to_json_kwargs(self) -> dict:
        pass


@register(Preprocessing, "select_channels")
class SelectChannels(Preprocessing):
    def __init__(self, channels, scalars=None):
        super().__init__()
        self.channels = channels
        self.scalars = scalars

    def call(self, x):
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

    def call(self, x, args=None):
        new_x = []
        for i in range(len(x)):
            one_item = []
            if self.channels:
                if self.z_range:
                    for z in self.z_range:
                        one_item.append([x[i][channel][z] for channel in self.channels])
                else:
                    for z in range(len(x[i][0])):
                        one_item.append([x[i][channel][z] for channel in self.channels])
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
