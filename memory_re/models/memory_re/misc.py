from typing import Tuple

import torch
from jerex import util
from sklearn.cluster import AgglomerativeClustering


def create_clusters(
    coref_clf: torch.Tensor, valid_mention_sample_masks: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = coref_clf.shape[0]

    batch_clusters = []
    batch_clusters_sample_masks = []

    for i in range(0, batch_size):
        clusters: list[list[int]] | None = None
        similarities = coref_clf[i]
        max_valid_mention: int = valid_mention_sample_masks[i].sum().int()  # type: ignore

        if max_valid_mention == 1:
            clusters = [[0]]
        elif max_valid_mention > 1:
            similarities = similarities[:max_valid_mention, :max_valid_mention]
            distances = 1 - torch.sigmoid(similarities).cpu().numpy()
            agg_clustering = AgglomerativeClustering(
                n_clusters=None,
                affinity='precomputed',
                linkage='complete',
                distance_threshold=1 - threshold,
            )
            assignments = agg_clustering.fit_predict(distances)
            mx = max(assignments)
            clusters = [[] for _ in range(mx + 1)]
            for mention_idx, cluster_index in enumerate(assignments):
                clusters[cluster_index].append(mention_idx)  # map back

        # -> tensors
        if clusters:
            batch_clusters.append(
                util.padded_stack([torch.tensor(list(c), dtype=torch.long) for c in clusters])
            )
            sample_mask = util.padded_stack(
                [torch.ones([len(c)], dtype=torch.bool) for c in clusters]
            )
            batch_clusters_sample_masks.append(sample_mask)
        else:
            batch_clusters.append(torch.zeros([1, 1], dtype=torch.long))
            batch_clusters_sample_masks.append(torch.zeros([1, 1], dtype=torch.bool))

    return util.padded_stack(batch_clusters).to(coref_clf.device), util.padded_stack(
        batch_clusters_sample_masks
    ).to(coref_clf.device)


def reindex_clusters(
    clusters: torch.Tensor,
    clusters_sample_masks: torch.Tensor,
    valid_mentions: torch.Tensor,
) -> torch.Tensor:
    if clusters_sample_masks.sum() == 0:
        return clusters

    batch_size = clusters.shape[0]
    batch_clusters = []

    for b in range(0, batch_size):
        _i = clusters[b]
        _clusters = valid_mentions[b][_i]
        _clusters[~clusters_sample_masks[b]] = 0
        batch_clusters.append(_clusters)

    return torch.stack(batch_clusters, dim=0)


def get_valid_mentions(
    mention_clf: torch.Tensor, mention_sample_masks: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = mention_clf.shape[0]
    mention_clf = (torch.sigmoid(mention_clf) >= threshold).float() * mention_sample_masks

    valid_mentions = []
    valid_mentions_masks = []
    for i in range(batch_size):
        non_zero_indices = mention_clf[i].nonzero().view(-1)
        valid_mentions.append(non_zero_indices)
        valid_mentions_masks.append(torch.ones_like(non_zero_indices))

    batch_valid_mentions = util.padded_stack(valid_mentions).to(mention_clf.device)
    batch_valid_mentions_masks = util.padded_stack(valid_mentions_masks).to(mention_clf.device)

    return mention_clf, batch_valid_mentions, batch_valid_mentions_masks
