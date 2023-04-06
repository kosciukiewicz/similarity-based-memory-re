from typing import Tuple

import jerex.evaluation.scoring

from memory_re.data.datasets.entities import Document


def convert_example(
    doc: Document,
    gt_mentions: list[Tuple],
    pred_mentions: list[Tuple],
    gt_clusters: list[Tuple],
    pred_clusters: list[Tuple],
    gt_entities: list[Tuple],
    pred_entities: list[Tuple],
    gt_relations: list[Tuple],
    pred_relations: list[Tuple],
):
    tokens = [t.phrase for t in doc.tokens]
    mention_tmp_args = _get_tp_fn_fp(gt_mentions, pred_mentions, tokens, _mention_to_html)
    cluster_tmp_args = _get_tp_fn_fp(gt_clusters, pred_clusters, tokens, _cluster_to_html)

    entity_tmp_args = _get_tp_fn_fp(gt_entities, pred_entities, tokens, _entity_to_html, type_idx=1)
    relation_tmp_args = _get_tp_fn_fp(
        gt_relations, pred_relations, tokens, _rel_to_html, type_idx=2
    )

    text = " ".join(tokens)
    return {
        'document_id': doc.doc_id,
        'mentions': mention_tmp_args,
        'clusters': cluster_tmp_args,
        'entities': entity_tmp_args,
        'relations': relation_tmp_args,
        'text': text,
    }


def _get_tp_fn_fp(gt, pred, tokens, to_html, type_idx=None):
    if gt or pred:
        scores = jerex.evaluation.scoring.score_single(gt, pred, type_idx=type_idx)
    else:
        scores = dict(zip(jerex.evaluation.scoring.METRIC_LABELS, [100.0] * 6))

    union = []
    for s in gt:
        if s not in union:
            union.append(s)

    for s in pred:
        if s not in union:
            union.append(s)

    # true positives
    tp = []
    # false negatives
    fn = []
    # false positives
    fp = []

    for s in union:
        type_verbose = s[type_idx].verbose_name if type_idx is not None else None

        if s in gt:
            if s in pred:
                tp.append({'text': to_html(s, tokens), 'type': type_verbose, 'c': 'tp'})
            else:
                fn.append({'text': to_html(s, tokens), 'type': type_verbose, 'c': 'fn'})
        else:
            fp.append({'text': to_html(s, tokens), 'type': type_verbose, 'c': 'fp'})

    return {
        'results': tp + fp + fn,
        'counts': {'tp': len(tp), 'fp': len(fp), 'fn': len(fn)},
        'scores': scores,
    }


def _mention_to_html(mention: Tuple, tokens: list[str]):
    start, end = mention[:2]

    tag_start = ' <span class="mention">'

    ctx_before = " ".join(tokens[:start])
    m = " ".join(tokens[start:end])
    ctx_after = " ".join(tokens[end:])

    html = ctx_before + tag_start + m + '</span> ' + ctx_after
    return html


def _cluster_to_html(cluster: Tuple, tokens: list[str]):
    cluster_list = list(cluster)
    cluster_list = sorted(cluster_list)

    tag_start = ' <span class="mention">'
    html = ""

    last_end = None
    for mention in cluster_list:
        start, end = mention
        ctx_before = " ".join(tokens[last_end:start])
        m = " ".join(tokens[start:end])
        html += ctx_before + tag_start + m + '</span> '
        last_end = end

    html += " ".join(tokens[cluster_list[-1][1] :])
    return html


def _entity_to_html(entity: Tuple, tokens: list[str]):
    cluster, entity_type = entity
    cluster = list(cluster)
    cluster = sorted(cluster)

    tag_start = ' <span class="mention">'
    html = ""

    last_end = None
    for mention in cluster:
        start, end = mention
        ctx_before = " ".join(tokens[last_end:start])
        m = " ".join(tokens[start:end])
        html += ctx_before + tag_start + m + '</span> '
        last_end = end

    html += " ".join(tokens[cluster[-1][1] :])
    return html


def _rel_to_html(relation: Tuple, tokens: list[str]):
    head, tail, rel_type = relation

    mentions = []
    head_cluster, head_entity_type = head
    tail_cluster, tail_entity_type = tail
    head_cluster, tail_cluster = list(head_cluster), list(tail_cluster)
    for h in head_cluster:
        mentions.append((h[0], h[1], 'h'))
    for t in tail_cluster:
        mentions.append((t[0], t[1], 't'))

    mentions = sorted(mentions)

    head_tag = ' <span class="head"><span class="type">%s</span>' % head_entity_type
    tail_tag = ' <span class="tail"><span class="type">%s</span>' % tail_entity_type
    html = ""

    last_end = None
    for mention in mentions:
        start, end, h_or_t = mention
        ctx_before = " ".join(tokens[last_end:start])
        m = " ".join(tokens[start:end])
        html += ctx_before + (head_tag if h_or_t == 'h' else tail_tag) + m + '</span> '
        last_end = end

    html += " ".join(tokens[mentions[-1][1] :])
    return html
