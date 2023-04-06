from typing import Any

import torch
import torch.nn.functional as F
from jerex import util
from jerex.models import misc as jerex_misc
from jerex.models.modules.coreference_resolution import CoreferenceResolution
from jerex.util import batch_index
from torch import nn
from transformers import PreTrainedTokenizer

from memory_re.models.memory_re.misc import create_clusters, reindex_clusters


class EntityRepresentation(nn.Module):
    def __init__(self, prop_drop):
        super().__init__()
        self.dropout = nn.Dropout(prop_drop)

    def forward(self, mention_reprs, entities, entity_masks, *args, **kwargs):
        mention_clusters = util.batch_index(mention_reprs, entities)
        entity_masks = entity_masks.unsqueeze(-1)

        # max pool entity clusters
        m = (entity_masks == 0).float() * (-1e30)
        mention_spans_pool = mention_clusters + m
        entity_reprs = mention_spans_pool.max(dim=2)[0]
        entity_reprs = self.dropout(entity_reprs)

        return entity_reprs


class CosineSimilarityCoreferenceResolution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()
        self._repr_size = hidden_size
        self._linear = nn.Linear(self._repr_size, self._repr_size)

    def get_mention_coref_representation(self, mention_reprs: torch.Tensor) -> torch.Tensor:
        return self._linear(mention_reprs[:, :, : self._repr_size])

    def forward(
        self, mention_reprs: torch.Tensor, valid_mentions: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        valid_mentions_reprs = batch_index(mention_reprs, valid_mentions)

        coref_mention_reprs = self.get_mention_coref_representation(valid_mentions_reprs)
        norm = coref_mention_reprs / coref_mention_reprs.norm(dim=2)[:, :, None]
        clf = torch.einsum("bnk,bkm->bnm", norm, norm.transpose(1, 2))  # n == m

        return {'coref_clf': clf, 'valid_mention_coref_reprs': coref_mention_reprs}

    def predict(
        self,
        mention_reprs: torch.Tensor,
        valid_mentions: torch.Tensor,
        valid_mention_sample_masks: torch.Tensor,
        threshold: float,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        coref_output = self(
            mention_reprs=mention_reprs,
            valid_mentions=valid_mentions,
        )
        coref_clf = coref_output['coref_clf']
        # create clusters
        clusters, clusters_sample_masks = create_clusters(
            coref_clf=coref_clf,
            valid_mention_sample_masks=valid_mention_sample_masks,
            threshold=threshold,
        )
        clusters = reindex_clusters(clusters, clusters_sample_masks, valid_mentions)

        return {
            'coref_clf': coref_clf,
            'clusters': clusters,
            'clusters_sample_masks': clusters_sample_masks,
        }


class ClassificationCoreferenceResolution(CoreferenceResolution):
    def __init__(
        self,
        hidden_size,
        meta_embedding_size,
        ed_embeddings_count,
        prop_drop,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__(hidden_size, meta_embedding_size, ed_embeddings_count, prop_drop)
        self._tokenizer = tokenizer

    def get_mention_coref_representation(self, mention_reprs: torch.Tensor) -> torch.Tensor:
        return mention_reprs

    def forward(
        self,
        mention_reprs: torch.Tensor,
        coref_mention_pairs: torch.Tensor,
        coref_eds: torch.Tensor,
        max_pairs: int | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        coref_clf = super().forward(
            mention_reprs=mention_reprs,
            coref_mention_pairs=coref_mention_pairs,
            coref_eds=coref_eds,
            max_pairs=max_pairs,
        )

        return {
            'coref_clf': coref_clf,
        }

    def predict(
        self,
        mention_reprs: torch.Tensor,
        valid_mention_masks: torch.Tensor,
        mention_spans: torch.Tensor,
        encodings: torch.Tensor,
        threshold: float,
        max_pairs: int | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        (
            coref_mention_pairs,
            coref_mention_eds,
            coref_sample_masks,
        ) = jerex_misc.create_coref_mention_pairs(
            valid_mention_masks, mention_spans, encodings, self._tokenizer
        )
        coref_clf = super().forward(
            mention_reprs=mention_reprs,
            coref_mention_pairs=coref_mention_pairs,
            coref_eds=coref_mention_eds,
            max_pairs=max_pairs,
        )

        clusters, clusters_sample_masks = jerex_misc.create_clusters(
            coref_clf=coref_clf,
            mention_pairs=coref_mention_pairs,
            pair_sample_mask=coref_sample_masks,
            valid_mentions=valid_mention_masks,
            threshold=threshold,
        )

        return {
            'coref_clf': coref_clf,
            'clusters': clusters,
            'clusters_sample_masks': clusters_sample_masks,
        }


class GatedMILAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int):
        super().__init__()

        self.attention_V = nn.Sequential(nn.Linear(hidden_size, attention_dim), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(hidden_size, attention_dim), nn.Sigmoid())

        self.attention_weights = nn.Linear(attention_dim, 1)

    def forward(self, _input: torch.Tensor, input_masks: torch.Tensor) -> torch.Tensor:
        a_V = self.attention_V(_input)
        a_U = self.attention_U(_input)
        a = self.attention_weights(a_V * a_U)
        a[~input_masks.bool()] = -1e25
        a = torch.nn.functional.softmax(a, dim=2)
        return a


class MILAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, _input: torch.Tensor, input_masks: torch.Tensor) -> torch.Tensor:
        # max pool entity clusters
        a = self.attention(_input)
        a[~input_masks.bool()] = -1e25
        a = torch.nn.functional.softmax(a, dim=2)
        return a


class MILAttentionEntityRepresentation(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        relation_memory_size: int,
        mil_attention_size: int,
        prop_drop: float = 0.1,
    ):
        super().__init__()

        self.attention = GatedMILAttention(
            hidden_size=hidden_size, attention_dim=mil_attention_size
        )
        self._relation_memory_read_linear = torch.nn.Linear(relation_memory_size, hidden_size)
        self._output_linear = torch.nn.Linear(hidden_size + relation_memory_size, hidden_size)
        self.dropout = nn.Dropout(prop_drop)

    def forward(
        self,
        mention_reprs: torch.Tensor,
        entities: torch.Tensor,
        entity_masks: torch.Tensor,
        relation_memory: torch.Tensor,
        *args,
        **kwargs,
    ):
        mention_clusters = util.batch_index(mention_reprs, entities)
        entity_masks = entity_masks.unsqueeze(-1)

        a = self.attention(_input=mention_clusters, input_masks=entity_masks)
        entity_reprs = a * mention_clusters
        entity_reprs = entity_reprs.sum(dim=2)

        s = self._relation_memory_read_linear(relation_memory.detach())
        s = F.linear(entity_reprs, s)
        s = F.linear(s, relation_memory.T)

        entity_reprs = torch.cat([entity_reprs, s], dim=-1)
        return self.dropout(self._output_linear(entity_reprs))


class EntityPairRepresentationGlobal(nn.Module):
    def __init__(self, hidden_size, prop_drop, entity_memory_size: int):
        super().__init__()

        self.entity_pair_linear = nn.Linear(hidden_size * 2 + entity_memory_size * 2, hidden_size)
        self.dropout = nn.Dropout(prop_drop)

    def forward(self, entity_reprs, rel_entity_types, pairs, entity_memory):
        rel_entity_types = F.embedding(rel_entity_types, entity_memory.detach())
        batch_size = pairs.shape[0]

        entity_pairs = util.batch_index(entity_reprs, pairs)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        rel_entity_types = rel_entity_types.view(
            rel_entity_types.shape[0], rel_entity_types.shape[1], -1
        )
        entity_pair_repr = self.entity_pair_linear(
            torch.cat([entity_pairs, rel_entity_types], dim=2)
        )
        entity_pair_repr = self.dropout(torch.relu(entity_pair_repr))

        return entity_pair_repr


class EntityPairRepresentationMultiInstance(nn.Module):
    def __init__(
        self,
        hidden_size,
        entity_types,
        entity_memory_size: int,
        meta_embedding_size,
        token_dist_embeddings_count,
        sentence_dist_embeddings_count,
        prop_drop,
    ):
        super().__init__()

        self.pair_linear = nn.Linear(hidden_size * 5 + 2 * meta_embedding_size, hidden_size)
        self.rel_linear = nn.Linear(hidden_size + 2 * entity_memory_size, hidden_size)

        self.token_distance_embeddings = nn.Embedding(
            token_dist_embeddings_count, meta_embedding_size
        )
        self.sentence_distance_embeddings = nn.Embedding(
            sentence_dist_embeddings_count, meta_embedding_size
        )
        self.entity_type_embeddings = nn.Embedding(entity_types, meta_embedding_size)

        self.dropout = nn.Dropout(prop_drop)

    def forward(
        self,
        pairs,
        entity_reprs,
        h,
        mention_reprs,
        entity_memory,
        rel_entity_pair_mp,
        rel_mention_pair_ep,
        rel_mention_pairs,
        rel_ctx_masks,
        rel_pair_masks,
        rel_token_distances,
        rel_sentence_distances,
        rel_entity_types,
        max_pairs=None,
    ):
        batch_size = pairs.shape[0]

        entity_pairs = util.batch_index(entity_reprs, pairs)
        entity_pair_reprs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        hidden_size = h.shape[-1]

        # relations
        # obtain relation logits
        # chunk processing to reduce memory usage
        max_pairs = max_pairs if max_pairs is not None else rel_mention_pairs.shape[1]
        rel_mention_pair_reprs = torch.zeros(
            [batch_size, rel_mention_pairs.shape[1], hidden_size]
        ).to(self._device)
        h = h.unsqueeze(1)

        for i in range(0, rel_mention_pairs.shape[1], max_pairs):
            # classify relation candidates
            chunk_rel_mention_pair_ep = rel_mention_pair_ep[:, i : i + max_pairs]
            chunk_rel_mention_pairs = rel_mention_pairs[:, i : i + max_pairs]
            chunk_rel_ctx_masks = rel_ctx_masks[:, i : i + max_pairs]
            chunk_rel_token_distances = rel_token_distances[:, i : i + max_pairs]
            chunk_rel_sentence_distances = rel_sentence_distances[:, i : i + max_pairs]
            chunk_h = h.expand(-1, chunk_rel_ctx_masks.shape[1], -1, -1)

            chunk_rel_logits = self._create_mention_pair_representations(
                entity_pair_reprs,
                chunk_rel_mention_pair_ep,
                chunk_rel_mention_pairs,
                chunk_rel_ctx_masks,
                chunk_rel_token_distances,
                chunk_rel_sentence_distances,
                mention_reprs,
                chunk_h,
            )

            rel_mention_pair_reprs[:, i : i + max_pairs, :] = chunk_rel_logits

        # classify relation candidates, get logits for each relation type per entity pair
        rel_clf = self._classify_relations(
            rel_mention_pair_reprs,
            rel_entity_pair_mp,
            rel_pair_masks,
            rel_entity_types,
            max_pairs=max_pairs,
            hidden_size=hidden_size,
            entity_memory=entity_memory,
        )

        return rel_clf

    def _create_mention_pair_representations(
        self,
        entity_pair_reprs,
        chunk_rel_mention_pair_ep,
        rel_mention_pairs,
        rel_ctx_masks,
        rel_token_distances,
        rel_sentence_distances,
        mention_reprs,
        h,
    ):
        rel_token_distances = self.token_distance_embeddings(rel_token_distances)
        rel_sentence_distances = self.sentence_distance_embeddings(rel_sentence_distances)

        rel_mention_pair_reprs = util.batch_index(mention_reprs, rel_mention_pairs)

        s = rel_mention_pair_reprs.shape
        rel_mention_pair_reprs = rel_mention_pair_reprs.view(s[0], s[1], -1)

        # ctx max pooling
        m = ((rel_ctx_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx, rel_ctx_indices = rel_ctx.max(dim=2)

        # set the context vector of neighboring or adjacent spans to zero
        rel_ctx[rel_ctx_masks.bool().any(-1) == 0] = 0

        entity_pair_reprs = util.batch_index(entity_pair_reprs, chunk_rel_mention_pair_ep)

        local_repr = torch.cat(
            [
                rel_ctx,
                rel_mention_pair_reprs,
                entity_pair_reprs,
                rel_token_distances,
                rel_sentence_distances,
            ],
            dim=2,
        )

        local_repr = self.dropout(self.pair_linear(local_repr))

        return local_repr

    def _classify_relations(
        self,
        rel_mention_pair_reprs: torch.Tensor,
        rel_entity_pair_mp: torch.Tensor,
        rel_pair_masks: torch.Tensor,
        rel_entity_types: torch.Tensor,
        max_pairs: int,
        hidden_size: int,
        entity_memory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = rel_mention_pair_reprs.shape[0]
        # Get representation
        rel_repr = torch.zeros([batch_size, rel_entity_pair_mp.shape[1], hidden_size]).to(
            self.rel_linear.weight.device
        )

        for i in range(0, rel_entity_pair_mp.shape[1], max_pairs):
            chunk_rel_entity_pair_mp = rel_entity_pair_mp[:, i : i + max_pairs]
            chunk_rel_pair_masks = rel_pair_masks[:, i : i + max_pairs]
            chunk_local_repr = util.batch_index(rel_mention_pair_reprs, chunk_rel_entity_pair_mp)
            chunk_rel_entity_types = rel_entity_types[:, i : i + max_pairs]
            chunk_local_repr += (chunk_rel_pair_masks.unsqueeze(-1) == 0).float() * (-1e30)
            chunk_local_repr = chunk_local_repr.max(dim=2)[0]

            chunk_rel_entity_types = F.embedding(chunk_rel_entity_types, entity_memory.detach())
            chunk_rel_entity_types = chunk_rel_entity_types.view(
                chunk_rel_entity_types.shape[0], chunk_rel_entity_types.shape[1], -1
            )

            chunk_rel_repr = torch.cat([chunk_local_repr, chunk_rel_entity_types], dim=2)
            chunk_rel_repr = torch.relu(self.rel_linear(chunk_rel_repr))
            rel_repr[:, i : i + max_pairs, :] = chunk_rel_repr

        rel_repr = self.dropout(rel_repr)

        return rel_repr

    @property
    def _device(self):
        return self.pair_linear.weight.device
