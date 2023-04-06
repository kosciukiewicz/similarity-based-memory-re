from abc import abstractmethod
from functools import reduce
from typing import Literal

import torch
from jerex.models.modules.coreference_resolution import CoreferenceResolution
from jerex.models.modules.mention_localization import MentionLocalization
from jerex.models.modules.mention_representation import MentionRepresentation
from jerex.util import batch_index
from pytorch_metric_learning.losses import ArcFaceLoss
from torch.nn import ModuleDict
from transformers import BertConfig, BertModel, BertPreTrainedModel, PreTrainedTokenizer

from memory_re.models.memory_re.memory_modules import (
    ArcFaceEntityClassification,
    ArcFaceMemoryModule,
    BilinearSimilarityEntityClassification,
    MatrixMemoryModule,
)
from memory_re.models.memory_re.memory_modules.base import (
    BaseMemoryModule,
    BaseMemoryReadingModule,
    EntityClassification,
    MemoryModule,
)
from memory_re.models.memory_re.memory_modules.bilinear import BilinearMemoryReadingModule
from memory_re.models.memory_re.memory_modules.cosine import CosineMemoryReadingModule
from memory_re.models.memory_re.misc import get_valid_mentions
from memory_re.models.memory_re.modules import (
    ClassificationCoreferenceResolution,
    CosineSimilarityCoreferenceResolution,
    EntityRepresentation,
)
from memory_re.training.memory_re_loss import MultiTaskJointLoss


class MemoryRE(BertPreTrainedModel):
    def __init__(
        self,
        config: BertConfig,
        tokenizer: PreTrainedTokenizer,
        criterion: MultiTaskJointLoss,
        relation_types: dict,
        entity_types: dict,
        prop_drop: float,
        meta_embedding_size: int,
        size_embeddings_count: int,
        ed_embeddings_count: int,
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
        memory_reading_module: Literal['Cosine', 'Bilinear'] = 'Bilinear',
        coref_resolution_type: Literal['Classification', 'CosineSimilarity'] = 'Classification',
        *args,
        **kwargs,
    ):
        self._tokenizer = tokenizer
        self._encoder_config = config
        super().__init__(config)

        # Transformer model
        self.bert = BertModel(config, add_pooling_layer=False)

        self._mention_threshold = mention_threshold
        self._coref_threshold = coref_threshold
        self._rel_threshold = rel_threshold

        self._use_entity_memory = use_entity_memory
        self._use_relation_memory = use_relation_memory
        self._memory_read_grad = memory_read_grad
        self._memory_reading_module = memory_reading_module

        self.entity_types = entity_types
        self.relation_types = relation_types
        self._coreference_resolution: CoreferenceResolution

        self.mention_representation = MentionRepresentation()

        self.mention_localization = MentionLocalization(
            config.hidden_size, meta_embedding_size, size_embeddings_count, prop_drop
        )

        self._init_coreference_resolution(
            coref_resolution_type=coref_resolution_type,
            meta_embedding_size=meta_embedding_size,
            ed_embeddings_count=ed_embeddings_count,
            prop_drop=prop_drop,
        )

        self.entity_representation = EntityRepresentation(prop_drop=prop_drop)

        self._memory_flow_modules = memory_flow_modules or []
        self._memory_read_modules = memory_read_modules or []

        self._init_memory_modules(
            criterion=criterion,
            entity_input_size=config.hidden_size,
            entity_types=entity_types,
            relation_types=relation_types,
            entity_memory_size=entity_memory_size or config.hidden_size,
            relation_memory_size=relation_memory_size or config.hidden_size,
        )

        self._init_memory_read_linear_weights()

    def _init_memory_modules(
        self,
        criterion: MultiTaskJointLoss,
        entity_input_size: int,
        entity_memory_size: int,
        entity_types: dict,
        relation_memory_size: int,
        relation_types: dict,
    ) -> None:
        self._init_entity_memory_module(
            criterion=criterion,
            entity_input_size=entity_input_size,
            entity_memory_size=entity_memory_size,
            entity_types=entity_types,
        )
        self._init_relation_memory_module(
            criterion=criterion,
            relation_memory_size=relation_memory_size,
            relation_types=relation_types,
        )

    def _init_entity_memory_module(
        self,
        criterion: MultiTaskJointLoss,
        entity_input_size: int,
        entity_memory_size: int,
        entity_types: dict,
    ) -> None:
        memory_module: BaseMemoryModule
        classifier: EntityClassification

        if isinstance(criterion.entity_criterion, ArcFaceLoss):
            memory_module = ArcFaceMemoryModule(criterion=criterion.entity_criterion)
            classifier = ArcFaceEntityClassification(criterion=criterion.entity_criterion)
        else:
            memory_module = MatrixMemoryModule(
                memory_size=entity_memory_size, memory_slots=len(entity_types)
            )
            classifier = BilinearSimilarityEntityClassification(
                input_size=entity_input_size, memory_module=memory_module
            )

        self._entity_classification = classifier
        self.entity_memory_module = MemoryModule(
            reading_module=self.get_memory_reading_module(
                self._memory_reading_module,
                repr_size=entity_input_size,
                memory_size=entity_memory_size,
            ),
            memory_module=memory_module,
        )

    def _init_relation_memory_module(
        self,
        criterion: MultiTaskJointLoss,
        relation_memory_size: int,
        relation_types: dict,
    ) -> None:
        memory_module: BaseMemoryModule

        if isinstance(criterion.relation_criterion, ArcFaceLoss):
            memory_module = ArcFaceMemoryModule(criterion=criterion.relation_criterion)
        else:
            memory_module = MatrixMemoryModule(
                memory_size=relation_memory_size, memory_slots=len(relation_types)
            )

        self.relation_memory_module = MemoryModule(
            reading_module=self.get_memory_reading_module(
                self._memory_reading_module,
                repr_size=self._encoder_config.hidden_size,
                memory_size=relation_memory_size,
            ),
            memory_module=memory_module,
        )

    def get_memory_reading_module(
        self,
        reading_module: str,
        repr_size: int,
        memory_size: int,
    ) -> BaseMemoryReadingModule:
        if reading_module == 'Bilinear':
            return BilinearMemoryReadingModule(
                repr_size=repr_size,
                memory_size=memory_size,
                memory_read_modules=self._memory_read_modules,
                memory_flow_modules=self._memory_flow_modules,
            )
        elif reading_module == 'Cosine':
            return CosineMemoryReadingModule()
        else:
            raise ValueError(f"Unsupported memory reading module {reading_module}")

    def _init_coreference_resolution(
        self,
        coref_resolution_type: Literal['Classification', 'CosineSimilarity'],
        meta_embedding_size: int,
        ed_embeddings_count: int,
        prop_drop: float,
    ) -> None:
        if coref_resolution_type == 'Classification':
            self._coreference_resolution = ClassificationCoreferenceResolution(
                hidden_size=self._encoder_config.hidden_size,
                meta_embedding_size=meta_embedding_size,
                ed_embeddings_count=ed_embeddings_count,
                prop_drop=prop_drop,
                tokenizer=self._tokenizer,
            )
        elif coref_resolution_type == 'CosineSimilarity':
            self._coreference_resolution = CosineSimilarityCoreferenceResolution(
                hidden_size=self._encoder_config.hidden_size
            )
        else:
            raise ValueError(f"Unsupported coreference resolution type {coref_resolution_type}")

    def _init_memory_read_linear_weights(self) -> None:
        size = (
            self._encoder_config.hidden_size
            + self.entity_memory_module.memory_size
            + self.relation_memory_module.memory_size
        )
        self.memory_read_linear_weights = ModuleDict(
            {
                module_name: torch.nn.Linear(size, self._encoder_config.hidden_size)
                for module_name in self._memory_read_modules
            }
        )

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(
                *args,
                **kwargs,
            )
        else:
            return self._forward_inference(
                *args,
                **kwargs,
            )

    def get_mention_coref_reprs(
        self,
        encodings: torch.Tensor,
        context_masks: torch.Tensor,
        mention_masks: torch.Tensor,
        mention_sample_masks: torch.Tensor,
        valid_mentions: torch.Tensor,
        use_memory: bool = True,
        max_spans: int = 100,
    ) -> torch.Tensor:
        h, mention_reprs, memory_modules_attentions = self._get_mention_reprs(
            encodings=encodings,
            context_masks=context_masks,
            mention_masks=mention_masks,
            mention_sample_masks=mention_sample_masks,
            use_memory=use_memory,
            max_spans=max_spans,
        )

        valid_mentions_reprs = batch_index(mention_reprs, valid_mentions)
        return self._coreference_resolution.get_mention_coref_representation(valid_mentions_reprs)

    @abstractmethod
    def _forward_train(self, *args, **kwargs):
        pass

    @abstractmethod
    def _forward_inference(self, *args, **kwargs):
        pass

    def _memory_read(
        self,
        reprs: torch.Tensor,
        reprs_mask: torch.Tensor,
        memory_module: str,
        use_entity_memory: bool = True,
        use_relation_memory: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # memory based mention representation
        reprs_list = [reprs]
        memory_attentions = {}

        if use_entity_memory:
            output, attentions = self.entity_memory_module.norm_read(
                input_reprs=reprs,
                input_mask=reprs_mask,
                module_name=memory_module,
                detach=not self._memory_read_grad,
            )
            memory_attentions['entity'] = attentions
            reprs_list.append(output)

        if use_relation_memory:
            output, attentions = self.relation_memory_module.norm_read(
                input_reprs=reprs,
                input_mask=reprs_mask,
                module_name=memory_module,
                detach=not self._memory_read_grad,
            )
            memory_attentions['relation'] = attentions
            reprs_list.append(output)

        reprs = torch.cat(reprs_list, dim=2)
        return self.memory_read_linear_weights[memory_module](reprs), memory_attentions

    def _memory_flow(
        self,
        reprs: torch.Tensor,
        reprs_mask: torch.Tensor,
        memory_module: str,
        use_entity_memory: bool = True,
        use_relation_memory: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # memory based mention representation
        reprs_list = [reprs]
        memory_attentions = {}
        if use_entity_memory:
            output, attentions = self.entity_memory_module.inverse_read(
                input_reprs=reprs,
                input_mask=reprs_mask,
                module_name=memory_module,
                detach=not self._memory_read_grad,
            )
            memory_attentions['entity'] = attentions
            reprs_list.append(output)

        if use_relation_memory:
            output, attentions = self.relation_memory_module.inverse_read(
                input_reprs=reprs,
                input_mask=reprs_mask,
                module_name=memory_module,
                detach=not self._memory_read_grad,
            )
            memory_attentions['relation'] = attentions
            reprs_list.append(output)

        reprs = reduce(
            lambda x, y: x + y,
            reprs_list,
        ) / len(reprs_list)
        return reprs, memory_attentions

    def _get_mention_reprs(
        self,
        encodings: torch.Tensor,
        context_masks: torch.Tensor,
        mention_masks: torch.Tensor,
        mention_sample_masks: torch.Tensor,
        max_spans: int | None = None,
        use_memory: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        context_masks = context_masks.float()
        mention_masks = mention_masks.float()
        use_entity_memory = self._use_entity_memory and use_memory
        use_relation_memory = self._use_relation_memory and use_memory
        memory_modules_attentions = {}

        bert_outputs = self.bert(input_ids=encodings, attention_mask=context_masks)  # type: ignore
        h = bert_outputs['last_hidden_state']

        if 'tokens' in self._memory_flow_modules:
            h, attentions = self._memory_flow(
                reprs=h,
                reprs_mask=context_masks,
                use_entity_memory=use_entity_memory,
                use_relation_memory=use_relation_memory,
                memory_module='tokens',
            )
            memory_modules_attentions['tokens_flow'] = attentions

        if 'tokens' in self._memory_read_modules and use_memory:
            h, attentions = self._memory_read(
                reprs=h,
                reprs_mask=context_masks,
                use_entity_memory=use_entity_memory,
                use_relation_memory=use_relation_memory,
                memory_module='tokens',
            )
            memory_modules_attentions['tokens_norm'] = attentions

        mention_reprs = self.mention_representation(h, mention_masks, max_spans=max_spans)

        if 'mentions' in self._memory_flow_modules:
            mention_reprs, attentions = self._memory_flow(
                reprs=mention_reprs,
                reprs_mask=mention_sample_masks,
                use_entity_memory=use_entity_memory,
                use_relation_memory=use_relation_memory,
                memory_module='mentions',
            )
            memory_modules_attentions['mentions_flow'] = attentions

        if 'mentions' in self._memory_read_modules and use_memory:
            mention_reprs, attentions = self._memory_read(
                reprs=mention_reprs,
                reprs_mask=mention_sample_masks,
                use_entity_memory=use_entity_memory,
                use_relation_memory=use_relation_memory,
                memory_module='mentions',
            )
            memory_modules_attentions['mentions_norm'] = attentions

        return h, mention_reprs, memory_modules_attentions

    def _forward_train_common(
        self,
        encodings: torch.Tensor,
        context_masks: torch.Tensor,
        mention_masks: torch.Tensor,
        mention_sizes: torch.Tensor,
        mention_sample_masks: torch.Tensor,
        entities: torch.Tensor,
        entity_masks: torch.Tensor,
        entity_sample_masks: torch.Tensor,
        coref_mention_pairs: torch.Tensor,
        valid_mentions: torch.Tensor,
        valid_mention_sample_masks: torch.Tensor,
        coref_eds,
        criterion: MultiTaskJointLoss,
        max_spans=None,
        max_coref_pairs=None,
        use_memory: bool = True,
    ):
        h, mention_reprs, memory_modules_attentions = self._get_mention_reprs(
            encodings=encodings,
            context_masks=context_masks,
            mention_masks=mention_masks,
            mention_sample_masks=mention_sample_masks,
            use_memory=use_memory,
            max_spans=max_spans,
        )
        mention_clf = self.mention_localization(mention_reprs, mention_sizes)
        coref_output = self._coreference_resolution(
            mention_reprs=mention_reprs,
            coref_eds=coref_eds,
            coref_mention_pairs=coref_mention_pairs,
            max_pairs=max_coref_pairs,
            valid_mentions=valid_mentions,
        )

        entity_reprs = self.entity_representation(
            mention_reprs,
            entities,
            entity_masks,
        )

        entity_clf = self._entity_classification(entity_reprs=entity_reprs, inference=False)

        return {
            'h': h,
            'mention_reprs': mention_reprs,
            'entity_reprs': entity_reprs,
            'mention_clf': mention_clf,
            'entity_clf': entity_clf,
            **coref_output,
        }

    def _forward_inference_common(
        self,
        encodings: torch.Tensor,
        context_masks: torch.Tensor,
        mention_masks: torch.Tensor,
        mention_sizes: torch.Tensor,
        mention_spans: torch.Tensor,
        mention_sample_masks: torch.Tensor,
        criterion: MultiTaskJointLoss,
        max_spans=None,
        max_coref_pairs=None,
        use_memory: bool = True,
    ):
        h, mention_reprs, memory_modules_attentions = self._get_mention_reprs(
            encodings=encodings,
            context_masks=context_masks,
            mention_masks=mention_masks,
            mention_sample_masks=mention_sample_masks,
            use_memory=use_memory,
            max_spans=max_spans,
        )

        # classify mentions
        mention_clf = self.mention_localization(mention_reprs, mention_sizes)
        valid_mentions_masks, valid_mentions, valid_mention_sample_masks = get_valid_mentions(
            mention_clf=mention_clf,
            mention_sample_masks=mention_sample_masks,
            threshold=self._mention_threshold,
        )

        # classify coreferences
        coref_output = self._coreference_resolution.predict(
            mention_reprs=mention_reprs,
            valid_mentions=valid_mentions,
            valid_mention_masks=valid_mentions_masks,
            valid_mention_sample_masks=valid_mention_sample_masks,
            mention_spans=mention_spans,
            encodings=encodings,
            max_pairs=max_coref_pairs,
            threshold=self._coref_threshold,
        )
        coref_clf = coref_output['coref_clf']
        clusters_sample_masks = coref_output['clusters_sample_masks']
        clusters = coref_output['clusters']

        entity_sample_masks = clusters_sample_masks.any(-1).float()

        # create entity representations
        entity_reprs = self.entity_representation(
            mention_reprs,
            clusters,
            clusters_sample_masks.float(),
        )

        # classify entities
        entity_clf = self._entity_classification(entity_reprs=entity_reprs, inference=True)

        return (
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
        )

    def _apply_thresholds(
        self,
        mention_clf: torch.Tensor,
        entity_clf: torch.Tensor,
        mention_sample_masks: torch.Tensor,
        entity_sample_masks: torch.Tensor,
    ):
        mention_clf = torch.sigmoid(mention_clf)
        mention_clf[mention_clf < self._mention_threshold] = 0
        mention_clf *= mention_sample_masks.float()

        entity_clf = torch.softmax(entity_clf, dim=-1)
        entity_clf *= entity_sample_masks.float().unsqueeze(-1)

        return mention_clf, entity_clf

    def _apply_rel_threshold(
        self,
        rel_clf: torch.Tensor,
        rel_sample_masks: torch.Tensor,
        criterion: MultiTaskJointLoss,
    ) -> torch.Tensor:
        if isinstance(criterion.relation_criterion, ArcFaceLoss):
            rel_clf = torch.softmax(rel_clf, dim=-1)
            max_values, _ = torch.max(rel_clf, dim=-1, keepdim=True)

            mask = rel_clf == max_values
            rel_clf[~mask] = 0
            rel_clf = rel_clf[:, :, 1:].contiguous()
        else:
            rel_clf = torch.sigmoid(rel_clf)
            rel_clf[rel_clf < self._rel_threshold] = 0

        rel_clf *= rel_sample_masks.float().unsqueeze(-1)

        return rel_clf
