import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
from numena.io.drive import Directory
from numena.io.image import imread_color
from numena.io.json import json_read

from kartezio.core.components import BaseGenotype
from kartezio.core.fitness import FitnessAP
from kartezio.easy import read_dataset
from kartezio.inference import KartezioModel
from kartezio.utils.viewer import KartezioViewer


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
    parser.add_argument("history", help="Path to the History", type=str)
    parser.add_argument("dataset", help="Path to the Dataset", type=str)
    parser.add_argument(
        "--prefix", help="Prefix to saved frames", type=str, default=None
    )
    parser.add_argument(
        "--crop",
        help="(x, y, w, h) tuple to crop images",
        type=int,
        nargs="+",
        default=None,
    )

    args = parser.parse_args()

    history_directory = Directory(args.history)
    model = KartezioModel(
        f"{history_directory._path}/elite.json", fitness=FitnessAP()
    )
    viewer = KartezioViewer(
        model._model.decoder.infos,
        model._model.decoder.library,
        model._model.decoder.endpoint,
    )
    dataset = read_dataset(dataset_path=args.dataset, indices=model.indices)
    cols_std = ["Parent", "Child", "Child"]
    cols_first = ["Child", "Child", "Child"]

    idx_to_frame = (
        list(range(1, 201)) + list(range(575, 626)) + list(range(1575, 1626))
    )
    frame_name_count = 1
    for i in idx_to_frame:
        frame_name = f"frame_{frame_name_count:04}.png"
        json_data = json_read(f"{history_directory._path}/G{i}.json")
        n_models = len(json_data["population"])
        imgs = []
        # dirty padding
        fitness = []
        for m in range(n_models):
            sequence = np.asarray(
                ast.literal_eval(json_data["population"][m]["sequence"])
            )
            genome = BaseGenotype(sequence=sequence)
            model._model.genome = genome
            p, f, t = model.eval(
                dataset, subset="train", reformat_x=reformat_x
            )
            fitness.append(1.0 - f)
            model_graph = viewer.get_graph(
                model._model.genome,
                inputs=["Tubulin", "DAPI"],
                outputs=["Mask", "Markers"],
            )
            model_graph.draw(path="tmp_graph_img.png")
            graph_image = imread_color("tmp_graph_img.png", rgb=True)
            imgs.append(graph_image)

        if i == 1:
            cols = cols_first
            title = "First Generation"
        else:
            cols = cols_std
            title = f"Generation {i}"

        fig, axes = plt.subplots(nrows=1, ncols=3, dpi=300, figsize=(8, 4))
        for j, (ax, col) in enumerate(zip(axes, cols)):
            ax.set_title(f"{col}\nf={fitness[j]:0.3f}", fontsize=7)
        fig.suptitle(title, fontsize=10)
        fig.subplots_adjust(wspace=0, hspace=0)

        index_image = 0
        for ax_col in axes:
            ax = ax_col
            ax.axis("off")
            ax.imshow(imgs[index_image])
            index_image += 1
        fig.tight_layout()
        plt.savefig(f"frames_graph/{frame_name}")
        fig.clear()
        frame_name_count += 1
