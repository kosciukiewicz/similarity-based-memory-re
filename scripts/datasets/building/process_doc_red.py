import argparse
from pathlib import Path
from typing import Any

from tqdm import tqdm

from memory_re.settings import DATASETS_DIR, RAW_DATA_DIR
from memory_re.utils.io import read_json_file, write_json_file

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--raw_data_dir', default=RAW_DATA_DIR / 'doc_red', type=str)
arg_parser.add_argument(
    '--destination_dir',
    default=DATASETS_DIR / 'relation_extraction' / 'doc_red',
    type=str,
)
arg_parser.add_argument(
    '--process_distant',
    default=True,
    type=bool,
)

args = arg_parser.parse_args()
base_raw_data_dir = Path(args.raw_data_dir)
base_destination_dir = Path(args.destination_dir)
types = read_json_file(base_raw_data_dir / 'types.json')


def _process_raw_file(raw_filepath: Path) -> list[dict[str, Any]]:
    raw_documents = read_json_file(raw_filepath)
    _processed_documents = []

    for doc in tqdm(raw_documents, desc=str(raw_filepath)):
        parsed_entities = []
        for entity in doc['vertexSet']:
            parsed_mentions = []
            for mention in entity:
                parsed_mentions.append(
                    {
                        'type': mention['type'],
                        'start': mention['pos'][0],
                        'end': mention['pos'][1],
                        'sent_id': mention['sent_id'],
                    }
                )

            parsed_entities.append(parsed_mentions)

        parsed_relations = []
        if 'labels' in doc:
            for relation in doc['labels']:
                parsed_relations.append(
                    {
                        "type": relation['r'],
                        "head": relation['h'],
                        "tail": relation['t'],
                        "evidence_sent_id": relation['evidence'],
                    }
                )

        _processed_documents.append(
            {'entities': parsed_entities, 'relations': parsed_relations, 'sents': doc['sents']}
        )

    return _processed_documents


base_destination_dir.mkdir(parents=True, exist_ok=True)
filenames_to_process = ['train_annotated', 'dev', 'test']

if args.process_distant:
    filenames_to_process += ['train_distant']

for filename in filenames_to_process:
    filepath = base_raw_data_dir / f"{filename}.json"
    destination_filepath = base_destination_dir / f"{filename}.json"

    processed_documents = _process_raw_file(raw_filepath=filepath)

    write_json_file(processed_documents, destination_filepath)

write_json_file(types, base_destination_dir / 'types.json')
