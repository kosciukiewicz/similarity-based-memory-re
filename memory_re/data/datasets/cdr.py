from memory_re.data.datasets.dataset import RelationExtractionDataset
from memory_re.settings import DATASETS_DIR


class CDRRelationExtractionDataset(RelationExtractionDataset):
    _BASE_DATASET_DIR = DATASETS_DIR / 'cdr'
    _SPLIT_FILEPATH_MAPPING = {
        'train': _BASE_DATASET_DIR / 'train.json',
        'val': _BASE_DATASET_DIR / 'dev.json',
        'test': _BASE_DATASET_DIR / 'test.json',
    }


class CDRFinalRelationExtractionDataset(RelationExtractionDataset):
    _BASE_DATASET_DIR = DATASETS_DIR / 'cdr_final'
    _SPLIT_FILEPATH_MAPPING = {
        'train': _BASE_DATASET_DIR / 'train.json',
        'val': _BASE_DATASET_DIR / 'test.json',
        'test': _BASE_DATASET_DIR / 'test.json',
    }
