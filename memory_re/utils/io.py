import json
import pickle as pkl
from pathlib import Path
from typing import Any, Type

import yaml
from omegaconf import DictKeyType


def load_txt_file(filepath: str | Path) -> str:
    with open(filepath) as f:
        return '\n'.join(f.readlines())


def read_json_file(filepath: str | Path) -> dict[Any, Any] | list[dict[Any, Any]]:
    with open(filepath) as f:
        data = f.read().replace('\n', '')
        return json.loads(data)


def read_yaml_file(filepath: str | Path) -> dict[str, Any]:
    with open(filepath, 'rb') as file:
        return yaml.safe_load(file)


def write_json_file(
    data: dict | list,
    filepath: str | Path,
    encoder_cls: Type[json.JSONEncoder] | None = None,
) -> None:
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=encoder_cls)


def save_json_file(data: dict[Any, Any], filepath: str | Path) -> None:
    with open(filepath, 'w') as f:
        json.dump(data, f)


def read_config(path: Path) -> dict[DictKeyType, Any]:
    """Returns config."""
    with path.open() as stream:
        return yaml.safe_load(stream)


def write_config(data: dict[DictKeyType, Any], path: Path) -> None:
    """Returns config."""
    with path.open(mode='w') as stream:
        return yaml.safe_dump(data, stream)


def write_pickle(data: Any, filepath: str | Path) -> None:
    with open(filepath, 'wb') as f:
        pkl.dump(data, f)


def read_pickle(filepath: str | Path) -> Any:
    with open(filepath, 'rb') as f:
        loaded_resource = pkl.load(f)

    return loaded_resource
