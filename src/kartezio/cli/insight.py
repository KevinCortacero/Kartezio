import argparse

import cv2
import numpy as np
from kartezio.core.registry import registry
from kartezio.easy import load_model, read_dataset
from kartezio.export import KartezioInsight
from kartezio.fitness import FitnessAP, FitnessIOU
from kartezio.inference import KartezioModel


def reformat_x(x):
    """
    Only keep the green (membrane) and red (nucleus) channels

    Parameters
    ----------
    x : original BGR set

    Returns
    -------
    Same set without the blue channel

    """
    new_x = []
    for i in range(len(x)):
        one_item = [x[i][1], x[i][2]]
        new_x.append(one_item)
    return new_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("core", help="Path to the genome", type=str)
    parser.add_argument("dataset", help="Path to the Dataset", type=str)
    parser.add_argument(
        "--prefix", help="Prefix to save images", type=str, default=None
    )
    parser.add_argument(
        "--crop",
        help="(x, y, w, h) tuple to crop images",
        type=int,
        nargs="+",
        default=None,
    )

    args = parser.parse_args()

    model = KartezioModel(args.model, registry.fitness.instantiate("AP"))
    insight = KartezioInsight(model._model.decoder)
    dataset = read_dataset(args.dataset, indices=model.indices)
    # p, f, t = core.eval(dataset, subset="test", reformat_x=reformat_x)
    # print(1.0 - f)
    # heatmap_color = cv2.applyColorMap(p[0]["labels"].astype(np.uint8)*5, cv2.COLORMAP_VIRIDIS)
    # cv2.imwrite("cellpose_out.png", heatmap_color)
    new_x = dataset.test_x  # reformat_x(dataset.test_x)
    for train_xi in [1]:  # range(len(dataset.train_x)):
        x = new_x[train_xi]
        insight.create_node_images(
            model._model.genome,
            x,
            prefix=f"{args.prefix}_train_{train_xi}",
            crop=args.crop,
        )
