import argparse
from pathlib import Path

from memory_re.settings import DATASETS_DIR
from memory_re.utils.io import read_json_file, write_json_file

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--dataset_dir', default=DATASETS_DIR / 'relation_extraction' / 'cdr', type=str
)
arg_parser.add_argument(
    '--destination_dir',
    default=DATASETS_DIR / 'relation_extraction' / 'cdr_final',
    type=str,
)

args = arg_parser.parse_args()
base_dataset_dir = Path(args.dataset_dir)
base_destination_dir = Path(args.destination_dir)

train_data: list[dict] = read_json_file(base_dataset_dir / 'train.json')  # type: ignore[assignment]
dev_data: list[dict] = read_json_file(base_dataset_dir / 'dev.json')  # type: ignore[assignment]
test_data: list[dict] = read_json_file(base_dataset_dir / 'test.json')  # type: ignore[assignment]
types_data = read_json_file(base_dataset_dir / 'types.json')

new_train_data = train_data + dev_data

base_destination_dir.mkdir(exist_ok=True, parents=True)
write_json_file(new_train_data, base_destination_dir / 'train.json')
write_json_file(test_data, base_destination_dir / 'test.json')
write_json_file(types_data, base_destination_dir / 'types.json')
