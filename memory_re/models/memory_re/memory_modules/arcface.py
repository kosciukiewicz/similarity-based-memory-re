import torch
from pytorch_metric_learning.losses import ArcFaceLoss

from memory_re.models.memory_re.memory_modules.base import (
    BaseMemoryModule,
    EntityClassification,
    RelationClassification,
)


class ArcFaceMemoryModule(BaseMemoryModule):
    def __init__(self, criterion: ArcFaceLoss) -> None:
        super().__init__()
        self._criterion = criterion

    @property
    def memory(self) -> torch.Tensor:
        return self._criterion.W.transpose(0, 1)


class ArcFaceEntityClassification(EntityClassification):
    def __init__(self, criterion: ArcFaceLoss):
        super().__init__()
        self._criterion = criterion

    def forward(self, entity_reprs: torch.Tensor, inference: bool = True):
        if inference:
            entity_reprs_ = entity_reprs.view(-1, entity_reprs.shape[-1])
            entity_clf = self._criterion.get_logits(entity_reprs_)
            return entity_clf.view(entity_reprs.shape[0], entity_reprs.shape[1], -1)
        else:
            return torch.zeros(entity_reprs.shape[0], self._criterion.W.data.shape[0])


class ArcFaceRelationClassification(RelationClassification):
    def __init__(self, criterion: ArcFaceLoss):
        super().__init__()
        self._criterion = criterion

    def forward(self, entity_pair_reprs: torch.Tensor, inference: bool = True):
        if inference:
            entity_pair_reprs_ = entity_pair_reprs.view(-1, entity_pair_reprs.shape[-1])
            rel_clf = self._criterion.get_cosine(entity_pair_reprs_)
            return rel_clf.view(entity_pair_reprs.shape[0], entity_pair_reprs.shape[1], -1)
        else:
            return torch.zeros(entity_pair_reprs.shape[0], self._criterion.W.data.shape[0])
