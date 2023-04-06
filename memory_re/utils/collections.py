from typing import Any


def merge_dicts(d1: dict, d2: dict) -> dict:
    for k in d2:
        if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
            merge_dicts(d1[k], d2[k])
        else:
            d1[k] = d2[k]

    return d1


def flatten(nested_list: list[list[Any]]):
    return [i for j in nested_list for i in j]
