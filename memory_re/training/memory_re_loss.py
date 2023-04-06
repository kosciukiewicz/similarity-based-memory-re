from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from omegaconf import DictConfig
from pytorch_metric_learning import reducers
from pytorch_metric_learning.losses import ArcFaceLoss
from torch import nn
from torch.nn import CrossEntropyLoss


class MultiTaskJointLoss(nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        num_entity_classes: int,
        num_relation_classes: int,
        coref_loss_config: DictConfig,
        entity_loss_config: DictConfig,
        relation_loss_config: DictConfig,
    ):
        super().__init__()
        self._mention_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._entity_criterion = self._build_entity_loss(
            num_classes=num_entity_classes, config=entity_loss_config
        )
        self._coref_criterion = self._build_coref_loss(coref_loss_config)
        self._rel_criterion = self._build_relation_loss(
            num_classes=num_relation_classes, config=relation_loss_config
        )

        self._entity_loss_config = entity_loss_config
        self._relation_loss_config = relation_loss_config
        self._coref_loss_config = coref_loss_config

    @property
    def entity_criterion(self) -> ArcFaceLoss | torch.nn.CrossEntropyLoss:
        return self._entity_criterion

    @property
    def relation_criterion(self) -> ArcFaceLoss | torch.nn.BCEWithLogitsLoss:
        return self._rel_criterion

    def _build_entity_loss(
        self, num_classes: int, config: DictConfig
    ) -> ArcFaceLoss | CrossEntropyLoss:
        if config.type == 'ArcFaceLoss':
            return ArcFaceLoss(
                num_classes=num_classes,
                embedding_size=config.embedding_size,
                margin=config.margin,
                scale=config.scale,
                reducer=reducers.DoNothingReducer(),
            )
        elif config.type == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported entity loss {config.type}")

    def _build_relation_loss(
        self, num_classes: int, config: DictConfig
    ) -> ArcFaceLoss | torch.nn.BCEWithLogitsLoss:
        if config.type == 'ArcFaceLoss':
            return ArcFaceLoss(
                num_classes=num_classes + 1,
                embedding_size=config.embedding_size,
                margin=config.margin,
                scale=config.scale,
                reducer=reducers.DoNothingReducer(),
            )
        elif config.type == 'BCEWithLogitsLoss':
            return torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported relation loss {config.type}")

    def _build_coref_loss(self, config: DictConfig) -> torch.nn.Module:
        if config.type == 'BCEWithLogitsLoss':
            return torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported coref loss {config.type}")

    @abstractmethod
    def _aggregate_loss(self, loss_list: list[torch.Tensor], **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def loss_weights(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def compute(
        self,
        mention_clf,
        entity_clf,
        coref_clf,
        rel_clf,
        mention_types,
        entity_types,
        coref_types,
        rel_types,
        mention_sample_masks,
        entity_sample_masks,
        coref_sample_masks,
        rel_sample_masks,
        **kwargs,
    ):
        loss_dict = {}

        losses = []

        # mention loss
        mention_clf = mention_clf.view(-1)
        mention_types = mention_types.view(-1).float()
        mention_sample_masks = mention_sample_masks.view(-1).float()

        mention_loss = self._mention_criterion(mention_clf, mention_types)
        mention_loss = (mention_loss * mention_sample_masks).sum() / mention_sample_masks.sum()

        losses.append(mention_loss)
        loss_dict['mention_loss'] = mention_loss

        # coref loss
        coref_loss = self._get_coref_loss(
            coref_clf=coref_clf,
            coref_types=coref_types,
            coref_sample_masks=coref_sample_masks,
            **kwargs,
        )

        loss_dict['coref_loss'] = coref_loss
        losses.append(coref_loss)

        # entity loss
        entity_loss = self._get_entity_loss(
            entity_clf=entity_clf,
            entity_types=entity_types,
            entity_sample_masks=entity_sample_masks,
            **kwargs,
        )

        losses.append(entity_loss)
        loss_dict['entity_loss'] = entity_loss

        # relation loss
        rel_loss = self._get_relation_loss(
            rel_clf=rel_clf,
            rel_types=rel_types,
            rel_sample_mask=rel_sample_masks,
            **kwargs,
        )

        losses.append(rel_loss)
        loss_dict['rel_loss'] = rel_loss

        loss_dict['target_loss'] = torch.vstack(losses).sum()
        loss_dict['loss'] = self._aggregate_loss(losses, **kwargs)

        return loss_dict

    def _get_coref_loss(
        self,
        coref_clf: torch.Tensor,
        coref_types: torch.Tensor,
        coref_sample_masks: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        loss_mask = coref_sample_masks.view(-1).float()
        coref_count = loss_mask.sum()

        if coref_count.item() == 0:
            return torch.tensor(0.0, device=coref_clf.device)

        coref_loss = self._get_bce_coref_loss(
            coref_clf=coref_clf,
            coref_types=coref_types,
        )

        return (coref_loss * loss_mask).sum() / coref_count.sum()

    def _get_entity_loss(
        self,
        entity_clf: torch.Tensor,
        entity_types: torch.Tensor,
        entity_sample_masks: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        loss_mask = entity_sample_masks.view(-1).float()
        entity_count = loss_mask.sum()

        if entity_count.item() == 0:
            return torch.tensor(0.0, device=entity_clf.device)

        if isinstance(self._entity_criterion, ArcFaceLoss):
            entity_loss = self._get_arcface_loss(
                criterion=self._entity_criterion,
                embeddings=kwargs['entity_reprs'],
                types=entity_types,
            )
        else:
            entity_clf = entity_clf.view(-1, entity_clf.shape[-1])
            entity_types = entity_types.view(-1)
            sample_entity_loss = self._entity_criterion(entity_clf, entity_types)
            entity_loss = (sample_entity_loss * loss_mask).sum() / entity_count

        return (entity_loss * loss_mask).sum() / entity_count.sum()

    def _get_relation_loss(
        self,
        rel_clf: torch.Tensor,
        rel_types: torch.Tensor,
        rel_sample_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        loss_mask = rel_sample_mask.view(-1).float()
        rel_count = loss_mask.sum()

        if rel_count.item() == 0:
            return torch.tensor(0.0, device=rel_clf.device)

        if isinstance(self._rel_criterion, ArcFaceLoss):
            rel_types_ = torch.zeros(
                (rel_types.shape[0], rel_types.shape[1], rel_types.shape[2] + 1), dtype=torch.long
            )
            rel_types_[:, :, 1:] = rel_types
            rel_types_ = torch.argmax(rel_types_, dim=-1)

            rel_loss = self._get_arcface_loss(
                criterion=self._rel_criterion,
                embeddings=kwargs['rel_reprs'],
                types=rel_types_,
            )
        else:
            rel_clf = rel_clf.view(-1, rel_clf.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_clf, rel_types.float())
            rel_loss = rel_loss.sum(-1)

        return (rel_loss * loss_mask).sum() / rel_count.sum()

    def _get_arcface_loss(
        self,
        criterion: ArcFaceLoss,
        embeddings: torch.Tensor,
        types: torch.Tensor,
    ):
        embeddings_ = embeddings.view(-1, embeddings.shape[-1])
        types_ = types.view(-1)
        return criterion(embeddings=embeddings_, labels=types_)['loss']['losses']

    def _get_bce_coref_loss(
        self,
        coref_clf: torch.Tensor,
        coref_types: torch.Tensor,
    ) -> torch.Tensor:
        coref_clf = coref_clf.view(-1)
        coref_types = coref_types.view(-1).float()
        return self._coref_criterion(coref_clf, coref_types)


class JointLoss(MultiTaskJointLoss):
    def __init__(
        self,
        num_entity_classes: int,
        num_relation_classes: int,
        coref_loss_config: DictConfig,
        entity_loss_config: DictConfig,
        relation_loss_config: DictConfig,
        task_weights=None,
    ):
        super().__init__(
            num_entity_classes=num_entity_classes,
            num_relation_classes=num_relation_classes,
            coref_loss_config=coref_loss_config,
            entity_loss_config=entity_loss_config,
            relation_loss_config=relation_loss_config,
        )
        self._task_weights = task_weights if task_weights else [1, 1, 1, 1]

    def _aggregate_loss(self, loss_list: list[torch.Tensor], **kwargs) -> torch.Tensor:
        return sum([task_loss * weight for task_loss, weight in zip(loss_list, self._task_weights)])

    @property
    def loss_weights(self) -> dict[str, torch.Tensor]:
        return {
            f'{k}_weight': v
            for k, v in zip(['mention', 'coref', 'entity', 'relation'], self._task_weights)
        }


def create_criterion(
    num_entity_classes: int,
    num_relation_classes: int,
    loss_config: DictConfig,
) -> MultiTaskJointLoss:
    match loss_config.type:
        case 'JerexJointLoss':
            return JointLoss(  # type: ignore
                num_entity_classes=num_entity_classes,
                num_relation_classes=num_relation_classes,
                coref_loss_config=loss_config.coref,
                entity_loss_config=loss_config.entity,
                relation_loss_config=loss_config.relation,
                task_weights=[
                    loss_config.weights.get(k, 1.0)
                    for k in ['mention', 'coref', 'entity', 'relation']
                ],
            )
        case loss_type:
            raise ValueError(f"{loss_type} not supported")
