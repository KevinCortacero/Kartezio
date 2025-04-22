import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from numena.io.drive import Directory
from numena.io.image import imread_color
from numena.io.json import json_read

from kartezio.core.components import BaseGenotype
from kartezio.core.fitness import FitnessAP
from kartezio.data import read_dataset
from kartezio.inference import KartezioModel
from kartezio.preprocessing import SelectChannels
from kartezio.utils.viewer import KartezioViewer

CHANNELS = [1, 2]
preprocessing = SelectChannels(CHANNELS)


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
    cols_std = ["Original Image", "Annotations", "Parent", "Child", "Child"]
    cols_first = ["Original Image", "Annotations", "Child", "Child", "Child"]
    last_fitness = 0
    idx_to_frame = (
        [1] * 10
        + list(range(1, 11))
        + list(range(10, 101, 5))
        + list(range(100, 501, 10))
        + list(range(500, 20001, 500))
        + [20000] * 10
    )
    print(len(idx_to_frame))
    frame_name_count = 1
    all_fitness = []
    all_indices = []
    last_graph = None
    for i in idx_to_frame:
        frame_name = f"frame_{frame_name_count:04}.png"
        json_data = json_read(f"{history_directory._path}/G{i}.json")
        n_models = len(json_data["population"])
        n_images = len(dataset.train_x)
        imgs = []
        fitness = []
        for img_idx in range(n_images):
            imgs.append(dataset.train_v[img_idx])
            imgs.append(dataset.train_y[img_idx][0])
            for m in range(n_models):
                sequence = np.asarray(
                    ast.literal_eval(json_data["population"][m]["sequence"])
                )
                genome = BaseGenotype(sequence=sequence)
                model._model.genome = genome
                p, f, t = model.eval(
                    dataset, subset="train", preprocessing=preprocessing
                )
                imgs.append(p[img_idx]["labels"])
                if img_idx == 0:
                    fitness.append(1.0 - f)
        best_fitness = max(fitness)
        all_fitness.append(best_fitness)
        all_indices.append(i)
        update_parent_graph = False
        if np.argmax(fitness) != 0:
            print("new parent: ", fitness)
            if last_graph is not None:
                update_parent_graph = True
        if last_fitness != best_fitness:
            update_parent_graph = True

        graphs = []
        for m in range(n_models):
            sequence = np.asarray(
                ast.literal_eval(json_data["population"][m]["sequence"])
            )
            genome = BaseGenotype(sequence=sequence)
            model._model.genome = genome

            """
            if m == 0 and not update_parent_graph and last_graph is not None:
                graphs.append(last_graph)
                continue
            """

            model_graph = viewer.get_graph(
                model._model.genome,
                inputs=["Phalloidin", "DAPI"],
                outputs=["Mask", "Markers"],
            )
            model_graph.draw(path=f"tmp_graph_img_{m}.png")
            graph_image = imread_color(f"tmp_graph_img_{m}.png", rgb=True)
            graphs.append(graph_image)
        last_graph = graphs[0]
        if i == 1:
            cols = cols_first
            title = "First Generation"
        else:
            cols = cols_std
            title = f"Generation {i}"
        rows = [f"Image {row}" for row in [1, 2, 3]]

        fig = plt.figure(dpi=300, figsize=(9, 5))
        grid = gridspec.GridSpec(4, 6)
        # grid.update(top=0.95, left=0.025, wspace=0.01, hspace=0.1, right=0.99, bottom=0.01)
        grid.update(left=0.05, right=0.99, bottom=0.01)
        ax1 = fig.add_subplot(grid[0, :2])
        ax1.set_title("Best AP Fitness over Generations", fontsize=7)
        ax1.set_xlim([1, i])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("Generations", fontsize=7)
        ax1.set_ylabel("AP", fontsize=7)
        ax1.tick_params(axis="both", which="major", labelsize=5)
        ax1.tick_params(axis="both", which="minor", labelsize=5)
        ax1.plot(all_indices, all_fitness)

        ax2 = fig.add_subplot(grid[0, 2])
        ax2.axis("off")
        ax2.set_title("Original Image A", fontsize=7)
        ax2.imshow(imgs[0])

        ax3 = fig.add_subplot(grid[0, 3])
        ax3.axis("off")
        ax3.set_title("Annotations A", fontsize=7)
        ax3.imshow(imgs[1])

        ax4 = fig.add_subplot(grid[0, 4])
        ax4.axis("off")
        ax4.set_title("Original Image B", fontsize=7)
        ax4.imshow(imgs[5])

        ax5 = fig.add_subplot(grid[0, 5])
        ax5.axis("off")
        ax5.set_title("Annotations B", fontsize=7)
        ax5.imshow(imgs[6])

        ax6 = fig.add_subplot(grid[1:3, :2])
        ax6.get_xaxis().set_ticks([])
        ax6.get_yaxis().set_ticks([])
        for spine in ax6.spines.values():
            spine.set_visible(False)
        ax6.set_xlabel(f"AP={fitness[0]:0.3f}", fontsize=7)
        ax6.set_title("Parent", fontsize=7)
        ax6.imshow(graphs[0])

        ax7 = fig.add_subplot(grid[3, 0])
        ax7.get_xaxis().set_ticks([])
        ax7.get_yaxis().set_ticks([])
        for spine in ax7.spines.values():
            spine.set_visible(False)
        ax7.set_xlabel(f"AP={fitness[1]:0.3f}", fontsize=7)
        ax7.set_title("Child 1", fontsize=7)
        ax7.imshow(graphs[1])

        ax8 = fig.add_subplot(grid[3, 1])
        ax8.get_xaxis().set_ticks([])
        ax8.get_yaxis().set_ticks([])
        for spine in ax8.spines.values():
            spine.set_visible(False)
        ax8.set_xlabel(f"AP={fitness[2]:0.3f}", fontsize=7)
        ax8.set_title("Child 2", fontsize=7)
        ax8.imshow(graphs[2])

        ax9 = fig.add_subplot(grid[1:3, 2:4])
        ax9.axis("off")
        ax9.set_title("Parent on Image A", fontsize=7)
        ax9.imshow(imgs[2])

        ax10 = fig.add_subplot(grid[3, 2])
        ax10.axis("off")
        ax10.set_title("Child 1 on Image A", fontsize=7)
        ax10.imshow(imgs[3])

        ax11 = fig.add_subplot(grid[3, 3])
        ax11.axis("off")
        ax11.set_title("Child 2 on Image A", fontsize=7)
        ax11.imshow(imgs[4])

        ax12 = fig.add_subplot(grid[1:3, 4:])
        ax12.axis("off")
        ax12.set_title("Parent on Image B", fontsize=7)
        ax12.imshow(imgs[7])

        ax13 = fig.add_subplot(grid[3, 4])
        ax13.axis("off")
        ax13.set_title("Child 1 on Image B", fontsize=7)
        ax13.imshow(imgs[8])

        ax14 = fig.add_subplot(grid[3, 5])
        ax14.axis("off")
        ax14.set_title("Child 2 on image B", fontsize=7)
        ax14.imshow(imgs[9])

        """
        for j, (ax, col) in enumerate(zip(axes[0], cols)):
            if j < 2:
                ax.set_title(col, fontsize=7)
            else:
                ax.set_title(f"{col}\nAP={fitness[j]:0.3f}", fontsize=7)
        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, fontsize=7)
        """

        fig.suptitle(title, fontsize=10)
        # fig.subplots_adjust(wspace=0, hspace=0)

        """
        index_image = 0
        for ax_col in axes:
            for ax_row in ax_col:
                ax = ax_row
                ax.imshow(imgs[index_image])
                ax.axis("off")
                index_image += 1
        """

        # fig.tight_layout()
        plt.savefig(f"frames/{frame_name}")
        fig.clear()
        plt.close(fig)
        frame_name_count += 1
        last_fitness = best_fitness


if __name__ == "__main__":
    main()
