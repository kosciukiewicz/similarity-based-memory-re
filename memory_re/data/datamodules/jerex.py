from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from jerex.sampling import sampling_classify, sampling_joint
from jerex.sampling.sampling_common import collate_fn_padding
from jerex.task_types import TaskType
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from memory_re.data.datasets import DATASETS, RelationExtractionDataset
from memory_re.data.datasets.entities import EntityType, RelationType, TokenizedDocument
from memory_re.utils.io import read_json_file


class JEREXDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer: BertTokenizer,
        task_type: str,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: int = 4,
        neg_mention_count: int = 200,
        neg_relation_count: int = 200,
        neg_coref_count: int = 200,
        max_span_size: int = 10,
        neg_mention_overlap_ratio: float = 0.5,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size or self._train_batch_size
        self._test_batch_size = test_batch_size or self._val_batch_size
        self._tokenizer = tokenizer
        self._task_type = task_type
        self._num_workers = num_workers
        self._neg_mention_count = neg_mention_count
        self._neg_relation_count = neg_relation_count
        self._neg_coref_count = neg_coref_count
        self._max_span_size = max_span_size
        self._neg_mention_overlap_ratio = neg_mention_overlap_ratio

        self.max_span_size = max_span_size

        self._train_dataset: RelationExtractionDataset | None = None
        self._val_dataset: RelationExtractionDataset | None = None
        self._test_dataset: RelationExtractionDataset | None = None

        self._entity_types = self._read_label2entity_type(
            DATASETS[self.dataset_name].types_filepath()
        )
        self._relation_types = self._read_label2relation_type(
            DATASETS[self.dataset_name].types_filepath()
        )

    @staticmethod
    def _read_label2entity_type(filepath: Path) -> dict[str, EntityType]:
        raw_types: dict[str, dict[str, Any]] = read_json_file(filepath)['entities']  # type: ignore
        entity_types: dict[str, EntityType] = {}
        for key, raw_type in raw_types.items():
            entity_types[key] = EntityType(
                index=len(entity_types), verbose_name=raw_type['verbose'], identifier=key
            )

        return entity_types

    @staticmethod
    def _read_label2relation_type(filepath: Path) -> dict[str, RelationType]:
        raw_types: dict[str, dict[str, Any]] = read_json_file(filepath)['relations']  # type: ignore
        relation_types: dict[str, RelationType] = {}
        for key, raw_type in raw_types.items():
            relation_types[key] = RelationType(
                index=len(relation_types),
                identifier=key,
                verbose_name=raw_type['verbose'],
                symmetric=raw_type['symmetric'],
            )

        return relation_types

    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None) -> None:
        if stage in ('fit', None):
            self._train_dataset = DATASETS[self.dataset_name](
                split='train', entity_types=self.entity_types, relation_types=self.relation_types
            )
            self._train_dataset.load_documents()

        if stage in ('fit', 'validate', None):
            self._val_dataset = DATASETS[self.dataset_name](
                split='val', entity_types=self.entity_types, relation_types=self.relation_types
            )
            self._val_dataset.load_documents()

        if stage == 'test':
            self._test_dataset = DATASETS[self.dataset_name](
                split='test', entity_types=self.entity_types, relation_types=self.relation_types
            )
            self._test_dataset.load_documents()

        self._tokenize_datasets()

    def _tokenize_datasets(self) -> None:
        if self._train_dataset is not None:
            self._train_dataset.tokenize(self._tokenizer)
        if self._val_dataset is not None:
            self._val_dataset.tokenize(self._tokenizer)
        if self._test_dataset is not None:
            self._test_dataset.tokenize(self._tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._num_workers,
            collate_fn=self._collate_train_sample,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            batch_size=self._val_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
            collate_fn=self._collate_eval_sample,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            batch_size=self._test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self._num_workers,
            collate_fn=self._collate_eval_sample,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    def _collate_train_sample(self, docs: list[TokenizedDocument]) -> dict[str, torch.Tensor]:
        batch = [self._process_train_sample(doc) for doc in docs]

        return collate_fn_padding(batch)

    def _collate_eval_sample(self, docs: list[TokenizedDocument]) -> dict[str, torch.Tensor]:
        batch = [self._process_eval_sample(doc) for doc in docs]

        return collate_fn_padding(batch)

    def _process_train_sample(self, doc: TokenizedDocument) -> dict[str, torch.Tensor]:
        if self._task_type == TaskType.JOINT:
            return sampling_joint.create_joint_train_sample(
                doc,
                self._neg_mention_count,
                self._neg_relation_count,
                self._neg_coref_count,
                self._max_span_size,
                self._neg_mention_overlap_ratio,
                len(self._relation_types),
            )
        elif self._task_type == TaskType.MENTION_LOCALIZATION:
            return sampling_classify.create_mention_classify_train_sample(
                doc,
                self._neg_mention_count,
                self._max_span_size,
                self._neg_mention_overlap_ratio,
            )
        elif self._task_type == TaskType.COREFERENCE_RESOLUTION:
            return sampling_classify.create_coref_classify_train_sample(doc, self._neg_coref_count)
        elif self._task_type == TaskType.ENTITY_CLASSIFICATION:
            return sampling_classify.create_entity_classify_sample(doc)
        elif self._task_type == TaskType.RELATION_CLASSIFICATION:
            return sampling_classify.create_rel_classify_train_sample(
                doc, self._neg_relation_count, len(self._relation_types)
            )
        else:
            raise Exception('Invalid task')

    def _process_eval_sample(self, doc: TokenizedDocument) -> dict[str, torch.Tensor]:
        if self._task_type == TaskType.JOINT:
            samples = sampling_joint.create_joint_inference_sample(doc, self._max_span_size)
        elif self._task_type == TaskType.MENTION_LOCALIZATION:
            samples = sampling_classify.create_mention_classify_inference_sample(
                doc, self._max_span_size
            )
        elif self._task_type == TaskType.COREFERENCE_RESOLUTION:
            samples = sampling_classify.create_coref_classify_inference_sample(doc)
        elif self._task_type == TaskType.ENTITY_CLASSIFICATION:
            samples = sampling_classify.create_entity_classify_sample(doc)
        elif self._task_type == TaskType.RELATION_CLASSIFICATION:
            samples = sampling_classify.create_rel_classify_inference_sample(doc)
        else:
            raise Exception('Invalid task')

        samples['doc_ids'] = torch.tensor(doc.doc_id, dtype=torch.long)
        return samples
