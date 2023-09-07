import colorsys

import cv2
import numpy as np
from numena.figure import Figure
from numena.image.color import rgb2bgr
from numena.image.contour import contours_find
from numena.image.drawing import draw_overlay


def save_prediction(
    filename, original, mask, color=[0, 255, 255], alpha=1.0, thickness=1
):
    mask_overlay = draw_overlay(
        original.copy(), mask, color=color, alpha=alpha, thickness=thickness
    )
    cv2.imwrite(filename, rgb2bgr(mask_overlay))


def plot_mask(original, mask, gt=None, color=[0, 255, 255], alpha=1.0, thickness=1):
    mask_overlay = draw_overlay(
        original.copy(), mask, color=color, alpha=alpha, thickness=thickness
    )
    offset = 0
    if gt is not None:
        gt_overlay = draw_overlay(
            original.copy(), gt, color=[0, 255, 0], alpha=alpha, thickness=thickness
        )
        fig = Figure(title="Mask Prediction", size=(12, 3))
        fig.create_panels(rows=1, cols=4)
        fig.set_panel(1, "Ground Truth", gt_overlay)
        offset = 1
    else:
        fig = Figure(title="Mask Prediction", size=(9, 3))
        fig.create_panels(rows=1, cols=3)
    fig.set_panel(0, "Original", original)
    fig.set_panel(1 + offset, "Mask", mask)
    fig.set_panel(2 + offset, "Mask Overlay", mask_overlay)


def plot_markers(original, markers, color=[255, 0, 0], use_centroid=True):
    fig = Figure(title="Markers Prediction", size=(12, 4))
    fig.create_panels(rows=1, cols=3)

    original_panel = fig.get_panel(0)
    original_panel.set_title("Original")
    original_panel.axis("off")
    original_panel.imshow(original)

    mask_panel = fig.get_panel(1)
    mask_panel.set_title("Markers")
    mask_panel.axis("off")
    mask_panel.imshow(markers, cmap="viridis")

    overlayed = original.copy()
    if use_centroid:
        cnts = contours_find(markers)
        for cnt in cnts:
            cnt_x = cnt[:, 0, 0]
            cnt_y = cnt[:, 0, 1]
            centroid_x = cnt_x.mean()
            centroid_y = cnt_y.mean()
            cv2.circle(overlayed, (int(centroid_x), int(centroid_y)), 10, color, -1)
    else:
        overlayed = draw_overlay(overlayed, markers, color=color)
    overlay_panel = fig.get_panel(2)
    overlay_panel.set_title("Overlayed")
    overlay_panel.axis("off")
    overlay_panel.imshow(overlayed)


def plot_watershed(original, mask, markers, labels, gt=None, filename=None):
    labels_overlay = original.copy()
    list_labels = np.unique(labels)
    N = len(list_labels)
    colors = np.array(
        [colorsys.hsv_to_rgb(*[x / N, 0.5, 0.5]) for x in range(1, N + 1)]
    )
    np.random.shuffle(colors)
    colors = colors * 255
    for i, label in enumerate(list_labels):
        if label == 0:
            continue
        label_color = colors[i].tolist()
        labels_overlay = draw_overlay(
            labels_overlay,
            (labels == label).astype(np.uint8),
            color=label_color,
            alpha=0.8,
        )
    offset = 0
    if gt is not None:
        gt_overlay = draw_overlay(original.copy(), gt, color=[0, 255, 0])
        fig = Figure(title="Watershed Prediction", size=(15, 3))
        fig.create_panels(rows=1, cols=5)
        fig.set_panel(1, "Ground Truth", gt_overlay)
        offset = 1
    else:
        fig = Figure(title="Watershed Prediction", size=(12, 3))
        fig.create_panels(rows=1, cols=4)
    fig.set_panel(0, "Original", original)
    fig.set_panel(1 + offset, "Mask", mask, cmap="viridis")
    fig.set_panel(2 + offset, "Markers", markers, cmap="viridis")
    fig.set_panel(3 + offset, "Labels", labels_overlay)

    if filename is not None:
        fig.save(filename)
        fig.close()
