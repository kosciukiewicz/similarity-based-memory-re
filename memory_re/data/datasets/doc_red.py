from memory_re.data.datasets.dataset import RelationExtractionDataset
from memory_re.settings import DATASETS_DIR


class DocREDRelationExtractionDataset(RelationExtractionDataset):
    _BASE_DATASET_DIR = DATASETS_DIR / 'doc_red'
    _SPLIT_FILEPATH_MAPPING = {
        'train': _BASE_DATASET_DIR / 'train_annotated.json',
        'val': _BASE_DATASET_DIR / 'dev.json',
        'test': _BASE_DATASET_DIR / 'test.json',
    }


class DistantDocREDRelationExtractionDataset(RelationExtractionDataset):
    _BASE_DATASET_DIR = DATASETS_DIR / 'doc_red'
    _SPLIT_FILEPATH_MAPPING = {
        'train': _BASE_DATASET_DIR / 'train_distant.json',
        'val': _BASE_DATASET_DIR / 'dev.json',
        'test': _BASE_DATASET_DIR / 'test.json',
    }


class JEREXDocREDJointRelationExtractionDataset(RelationExtractionDataset):
    _BASE_DATASET_DIR = DATASETS_DIR / 'jerex_doc_red_joint'
    _SPLIT_FILEPATH_MAPPING = {
        'train': _BASE_DATASET_DIR / 'train_joint.json',
        'val': _BASE_DATASET_DIR / 'dev_joint.json',
        'test': _BASE_DATASET_DIR / 'test_joint.json',
    }


class ReDocREDRelationExtractionDataset(RelationExtractionDataset):
    _BASE_DATASET_DIR = DATASETS_DIR / 're_doc_red'
    _SPLIT_FILEPATH_MAPPING = {
        'train': _BASE_DATASET_DIR / 'train_revised.json',
        'val': _BASE_DATASET_DIR / 'dev_revised.json',
        'test': _BASE_DATASET_DIR / 'test_revised.json',
    }


class RevisitDocREDRelationExtractionDataset(RelationExtractionDataset):
    _BASE_DATASET_DIR = DATASETS_DIR / 'revisit_doc_red'
    _SPLIT_FILEPATH_MAPPING = {
        'train': _BASE_DATASET_DIR / 'train_annotated.json',
        'val': _BASE_DATASET_DIR / 'valid_scratch.json',
        'test': _BASE_DATASET_DIR / 'valid_scratch.json',
    }
