import argparse

from kartezio.data.dataset import DatasetMeta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of your dataset", type=str)
    parser.add_argument("label_name", help="Name of one entity of your set", type=str)
    parser.add_argument(
        "--input_type",
        help="Type of your input files",
        type=str,
        choices=["image"],
        default="image",
    )
    parser.add_argument(
        "--label_type",
        help="Type of your label files",
        type=str,
        choices=["image", "csv"],
        default="csv",
    )
    parser.add_argument(
        "--input_format",
        help="Format of your input files",
        type=str,
        choices=["grayscale, rgb"],
        default="rgb",
    )
    parser.add_argument(
        "--label_format",
        help="Format of your label files",
        type=str,
        choices=["ellipse, labels"],
        default="ellipse",
    )
    parser.add_argument(
        "--scale", help="Scale of your label files", type=float, default=1.0
    )
    args = parser.parse_args()

    dataset_path = "./"
    DatasetMeta.write(
        dataset_path,
        args.name,
        args.input_type,
        args.input_format,
        args.label_type,
        args.label_format,
        args.label_name,
    )
    print("META.json generated!")
