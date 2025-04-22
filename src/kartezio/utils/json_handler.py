import json


def json_read(filepath) -> dict:
    """Read the JSON file with the given filepath.

    Parameters
    ----------
    filepath :

    Returns
    -------
    dict
        The JSON data of the given file as a dict or list.

    """
    with open(filepath, "rb") as json_file:
        json_data = json.load(json_file)
        return json_data


def json_write(filepath, json_data, indent=4):
    """Write the given json_data to the JSON file with the given filepath.

    Parameters
    ----------
    filepath :
    json_data :
    indent :
    """
    with open(filepath, "w") as json_file:
        json.dump(json_data, json_file, indent=indent)
