import argparse

from kartezio.easy import load_model
from kartezio.utils.viewer import KartezioViewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to the genome", type=str)
    parser.add_argument(
        "--inputs",
        help="List of the names of the inputs",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--outputs",
        help="List of the names of the outputs",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--filename", help="Name of the file", type=str, default="model_graph.png"
    )

    args = parser.parse_args()

    print(args.model)
    model = load_model(args.model)
    viewer = KartezioViewer(
        model.decoder.infos, model.decoder.library, model.decoder.endpoint
    )
    model_graph = viewer.get_graph(
        model.genome, inputs=args.inputs, outputs=args.outputs
    )
    model_graph.draw(args.filename)
