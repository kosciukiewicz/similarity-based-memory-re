import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from memory_re.settings import DATASETS_DIR, PROJECT_DIR, RAW_DATA_DIR
from memory_re.utils.io import read_yaml_file, write_json_file

sys.path.append(str(PROJECT_DIR / 'submodules' / 'edge_oriented_graph'))
import src.loader.DataLoader as EdgeOrientedGraphDataLoader  # noqa E402

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--raw_data_dir', default=RAW_DATA_DIR / 'cdr', type=str)
arg_parser.add_argument(
    '--destination_dir',
    default=DATASETS_DIR / 'relation_extraction' / 'cdr',
    type=str,
)

args = arg_parser.parse_args()
base_raw_data_dir = args.raw_data_dir
base_destination_dir = args.destination_dir

parameters = read_yaml_file(
    PROJECT_DIR / 'submodules' / 'edge_oriented_graph' / 'configs' / 'parameters_cdr.yaml'
)
parameters['embeds'] = None

entity_types = {}
relation_types = {}


def _process_raw_file(raw_filepath: Path) -> list[dict[str, Any]]:
    data_loader = EdgeOrientedGraphDataLoader(raw_filepath, parameters)
    data_loader.__call__()

    _processed_documents = []

    for (doc_id, sentences) in tqdm(data_loader.documents.items(), desc=str(raw_filepath)):
        entity2index: dict[int, int] = {}
        entity2sent_idx: dict[int, set[int]] = {}

        parsed_entities = []
        doc_entities = data_loader.entities[doc_id]
        doc_relations = data_loader.pairs[doc_id]
        sentences_start_index = np.array([len(sentence) for sentence in sentences])
        sentences_start_index = np.cumsum(sentences_start_index) - sentences_start_index

        for entity_id, entity in doc_entities.items():
            parsed_mentions = []
            entity_sentences = []
            for mention_start, mention_end, sent_id in zip(
                entity.mstart.split(':'), entity.mend.split(':'), entity.sentNo.split(':')
            ):
                if entity.type not in entity_types:
                    entity_types[entity.type] = dict(short=entity.type, verbose=entity.type)

                sent_id = int(sent_id)
                mention_start = int(mention_start) - sentences_start_index[sent_id]
                mention_end = int(mention_end) - sentences_start_index[sent_id]
                parsed_mentions.append(
                    dict(
                        type=entity.type,
                        start=int(mention_start),
                        end=int(mention_end),
                        sent_id=sent_id,
                    )
                )
                entity_sentences.append(int(sent_id))

            entity2index[entity_id] = len(entity2index)
            entity2sent_idx[entity_id] = set(entity_sentences)
            parsed_entities.append(parsed_mentions)

        parsed_relations = []
        for (head_entity_id, tail_entity_id), relation in doc_relations.items():
            if (tail_entity_id, head_entity_id) in doc_relations:
                print('symmetric? ', (head_entity_id, tail_entity_id))

            if relation.direction == 'R2L':
                (head_entity_id, tail_entity_id) = (tail_entity_id, head_entity_id)

            if relation.type not in relation_types:
                relation_types[relation.type] = dict(
                    short=relation.type, verbose=relation.type, symmetric=False
                )

            parsed_relations.append(
                dict(
                    type=relation.type,
                    head=entity2index[head_entity_id],
                    tail=entity2index[tail_entity_id],
                    evidence_sent_id=list(
                        entity2sent_idx[head_entity_id].intersection(
                            entity2sent_idx[tail_entity_id]
                        )
                    ),
                )
            )

        _processed_documents.append(
            {'entities': parsed_entities, 'relations': parsed_relations, 'sents': sentences}
        )

    return _processed_documents


for filename in ['train', 'dev', 'test']:
    filepath = base_raw_data_dir / f'{filename}_filter.data'
    destination_filepath = base_destination_dir / f"{filename}.json"

    processed_documents = _process_raw_file(filepath)

    write_json_file(processed_documents, destination_filepath)

write_json_file(
    dict(entities=entity_types, relations=relation_types), base_destination_dir / 'types.json'
)
