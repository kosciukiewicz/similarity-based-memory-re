from pathlib import Path

import torch
from jerex import util
from jerex.models import misc
from pytorch_metric_learning.losses import ArcFaceLoss
from transformers import BertConfig, PreTrainedTokenizer

from memory_re.models.memory_re.memory_modules import (
    ArcFaceRelationClassification,
    BilinearSimilarityRelationClassification,
)
from memory_re.models.memory_re.memory_modules.base import RelationClassification
from memory_re.models.memory_re.memory_re import MemoryRE
from memory_re.models.memory_re.modules import EntityPairRepresentationMultiInstance
from memory_re.training.memory_re_loss import MultiTaskJointLoss


class MemoryREMultiInstanceModel(MemoryRE):
    def __init__(
        self,
        config: BertConfig,
        criterion: MultiTaskJointLoss,
        tokenizer: PreTrainedTokenizer,
        relation_types: dict,
        entity_types: dict,
        prop_drop: float,
        meta_embedding_size: int,
        size_embeddings_count: int,
        ed_embeddings_count: int,
        token_dist_embeddings_count: int,
        sentence_dist_embeddings_count: int,
        mention_threshold: float,
        coref_threshold: float,
        rel_threshold: float,
        memory_flow_modules: list[str] | None,
        memory_read_modules: list[str] | None,
        use_entity_memory: bool,
        use_relation_memory: bool,
        memory_read_grad: bool,
        entity_memory_size: int | None,
        relation_memory_size: int | None,
        *args,
        **kwargs,
    ):
        super().__init__(
            config,
            tokenizer,
            criterion,
            relation_types,
            entity_types,
            prop_drop,
            meta_embedding_size,
            size_embeddings_count,
            ed_embeddings_count,
            mention_threshold,
            coref_threshold,
            rel_threshold,
            memory_flow_modules,
            memory_read_modules,
            use_entity_memory,
            use_relation_memory,
            memory_read_grad,
            entity_memory_size,
            relation_memory_size,
            *args,
            **kwargs,
        )

        self.entity_pair_representation = EntityPairRepresentationMultiInstance(
            hidden_size=config.hidden_size,
            entity_types=len(entity_types),
            meta_embedding_size=meta_embedding_size,
            token_dist_embeddings_count=token_dist_embeddings_count,
            sentence_dist_embeddings_count=sentence_dist_embeddings_count,
            entity_memory_size=self.entity_memory_module.memory_size,
            prop_drop=prop_drop,
        )
        self.relation_classification: RelationClassification

        if isinstance(criterion.relation_criterion, ArcFaceLoss):
            self.relation_classification = ArcFaceRelationClassification(
                criterion=criterion.relation_criterion
            )
        else:
            self.relation_classification = BilinearSimilarityRelationClassification(
                hidden_size=config.hidden_size, memory_module=self.relation_memory_module
            )

        # weight initialization
        self.init_weights()  # type: ignore

    def save(self, destination_dir: str | Path) -> None:
        pass

    def _forward_train(  # type: ignore
        self,
        encodings: torch.Tensor,
        context_masks: torch.Tensor,
        mention_masks: torch.Tensor,
        mention_sizes: torch.Tensor,
        mention_sample_masks: torch.Tensor,
        valid_mentions: torch.Tensor,
        valid_mention_sample_masks: torch.Tensor,
        entities: torch.Tensor,
        entity_masks: torch.Tensor,
        entity_sample_masks: torch.Tensor,
        coref_mention_pairs: torch.Tensor,
        rel_entity_pairs: torch.Tensor,
        rel_mention_pairs: torch.Tensor,
        rel_ctx_masks: torch.Tensor,
        rel_entity_pair_mp: torch.Tensor,
        rel_mention_pair_ep: torch.Tensor,
        rel_pair_masks: torch.Tensor,
        rel_token_distances: torch.Tensor,
        rel_sentence_distances: torch.Tensor,
        entity_types: torch.Tensor,
        coref_eds: torch.Tensor,
        criterion: MultiTaskJointLoss,
        use_memory: bool = True,
        max_spans: bool | None = None,
        max_coref_pairs: bool | None = None,
        max_rel_pairs: bool | None = None,
        *args,
        **kwargs,
    ):
        res = self._forward_train_common(
            encodings=encodings,
            context_masks=context_masks,
            mention_masks=mention_masks,
            mention_sizes=mention_sizes,
            mention_sample_masks=mention_sample_masks,
            valid_mentions=valid_mentions,
            valid_mention_sample_masks=valid_mention_sample_masks,
            entities=entities,
            entity_masks=entity_masks,
            entity_sample_masks=entity_sample_masks,
            coref_mention_pairs=coref_mention_pairs,
            coref_eds=coref_eds,
            max_spans=max_spans,
            max_coref_pairs=max_coref_pairs,
            use_memory=use_memory,
            criterion=criterion,
        )
        entity_reprs = res['entity_reprs']
        h = res['h']
        mention_reprs = res['mention_reprs']

        rel_entity_types = util.batch_index(
            entity_types,
            rel_entity_pairs,
        )

        entity_pair_reprs = self.entity_pair_representation(
            entity_reprs=entity_reprs,
            pairs=rel_entity_pairs,
            h=h,
            mention_reprs=mention_reprs,
            rel_entity_pair_mp=rel_entity_pair_mp,
            rel_mention_pair_ep=rel_mention_pair_ep,
            rel_mention_pairs=rel_mention_pairs,
            rel_ctx_masks=rel_ctx_masks,
            rel_pair_masks=rel_pair_masks,
            rel_token_distances=rel_token_distances,
            rel_sentence_distances=rel_sentence_distances,
            rel_entity_types=rel_entity_types,
            entity_memory=self.entity_memory_module.memory,
            max_pairs=max_rel_pairs,
        )

        rel_clf = self.relation_classification(entity_pair_reprs=entity_pair_reprs, inference=False)

        return dict(**res, rel_clf=rel_clf, rel_reprs=entity_pair_reprs)

    def _forward_inference(  # type: ignore
        self,
        encodings: torch.Tensor,
        context_masks: torch.Tensor,
        mention_masks: torch.Tensor,
        mention_sizes: torch.Tensor,
        mention_spans: torch.Tensor,
        mention_sample_masks: torch.Tensor,
        mention_sent_indices: torch.Tensor,
        mention_orig_spans: torch.Tensor,
        criterion: MultiTaskJointLoss,
        max_spans: bool | None = None,
        max_coref_pairs: bool | None = None,
        max_rel_pairs: bool | None = None,
        use_memory: bool = True,
        *args,
        **kwargs,
    ):
        res = self._forward_inference_common(
            encodings,
            context_masks,
            mention_masks,
            mention_sizes,
            mention_spans,
            mention_sample_masks,
            use_memory=use_memory,
            max_spans=max_spans,
            max_coref_pairs=max_coref_pairs,
            criterion=criterion,
        )
        (
            h,
            mention_reprs,
            entity_reprs,
            clusters,
            entity_sample_masks,
            clusters_sample_masks,
            mention_clf,
            entity_clf,
            coref_clf,
            memory_modules_attentions,
            valid_mentions_masks,
            valid_mentions,
            valid_mention_sample_masks,
        ) = res

        # create entity pairs
        (
            rel_entity_pair_mp,
            rel_mention_pair_ep,
            rel_entity_pairs,
            rel_mention_pairs,
            rel_ctx_masks,
            rel_token_distances,
            rel_sentence_distances,
            rel_mention_pair_masks,
        ) = misc.create_local_entity_pairs(
            clusters,
            clusters_sample_masks,
            mention_spans,
            mention_sent_indices,
            mention_orig_spans,
            context_masks.shape[-1],
        )
        rel_sample_masks = rel_mention_pair_masks.any(dim=-1)
        entity_types = entity_clf.argmax(dim=-1)
        rel_entity_types = util.batch_index(entity_types, rel_entity_pairs)

        entity_pair_reprs = self.entity_pair_representation(
            entity_reprs=entity_reprs,
            pairs=rel_entity_pairs,
            h=h,
            mention_reprs=mention_reprs,
            rel_entity_pair_mp=rel_entity_pair_mp,
            rel_mention_pair_ep=rel_mention_pair_ep,
            rel_mention_pairs=rel_mention_pairs,
            rel_ctx_masks=rel_ctx_masks,
            rel_pair_masks=rel_mention_pair_masks,
            rel_token_distances=rel_token_distances,
            rel_sentence_distances=rel_sentence_distances,
            rel_entity_types=rel_entity_types,
            entity_memory=self.entity_memory_module.memory,
            max_pairs=max_rel_pairs,
        )

        # thresholding and masking
        mention_clf, entity_clf = self._apply_thresholds(
            mention_clf,
            entity_clf,
            mention_sample_masks,
            entity_sample_masks,
        )

        # classify relations
        rel_clf = self.relation_classification(entity_pair_reprs=entity_pair_reprs, inference=True)

        rel_clf = self._apply_rel_threshold(
            rel_clf=rel_clf, rel_sample_masks=rel_sample_masks, criterion=criterion
        )

        return dict(
            mention_clf=mention_clf,
            coref_clf=coref_clf,
            entity_clf=entity_clf,
            rel_clf=rel_clf,
            clusters=clusters,
            clusters_sample_masks=clusters_sample_masks,
            rel_entity_pairs=rel_entity_pairs,
            memory_modules_attentions=memory_modules_attentions,
            valid_mentions=valid_mentions,
            valid_mention_sample_masks=valid_mention_sample_masks,
        )
