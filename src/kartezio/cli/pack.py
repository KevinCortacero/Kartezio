import argparse

from numena.io.drive import Directory

from kartezio.utils.io import pack_one_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", help="Path to the directory containing training workspaces.", type=str
    )
    args = parser.parse_args()
    directory = Directory(args.path)
    directories = []
    for elite in directory.ls("*/elite.json", ordered=True):
        directories.append(elite.parent)
    for d in directories:
        path = f"{args.path}/{d}"
        pack_one_directory(path)
