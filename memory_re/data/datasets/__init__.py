from typing import Type

from memory_re.data.datasets.cdr import (
    CDRFinalRelationExtractionDataset,
    CDRRelationExtractionDataset,
)
from memory_re.data.datasets.dataset import RelationExtractionDataset
from memory_re.data.datasets.doc_red import (
    DistantDocREDRelationExtractionDataset,
    DocREDRelationExtractionDataset,
    JEREXDocREDJointRelationExtractionDataset,
    ReDocREDRelationExtractionDataset,
    RevisitDocREDRelationExtractionDataset,
)

DATASETS: dict[str, Type[RelationExtractionDataset]] = {  # type: ignore
    dataset.__name__: dataset
    for dataset in (
        CDRRelationExtractionDataset,
        CDRFinalRelationExtractionDataset,
        DistantDocREDRelationExtractionDataset,
        DocREDRelationExtractionDataset,
        JEREXDocREDJointRelationExtractionDataset,
        RevisitDocREDRelationExtractionDataset,
        ReDocREDRelationExtractionDataset,
    )
}

__all__ = [
    'RevisitDocREDRelationExtractionDataset',
    'ReDocREDRelationExtractionDataset',
    'DistantDocREDRelationExtractionDataset',
    'DocREDRelationExtractionDataset',
    'JEREXDocREDJointRelationExtractionDataset',
    'RelationExtractionDataset',
    'DATASETS',
    'RelationExtractionDataset',
]
