from typing import Literal

import torch
from torch import nn
from transformers import BertConfig, BertTokenizer

from memory_re.models.memory_re.memory_re import MemoryRE
from memory_re.models.memory_re.memory_re_global import MemoryREGlobalModel
from memory_re.models.memory_re.memory_re_multi_instance import MemoryREMultiInstanceModel
from memory_re.training.memory_re_loss import MultiTaskJointLoss

_MODELS = {
    # joint models
    'multi_instance': MemoryREMultiInstanceModel,
    'global': MemoryREGlobalModel,
}


def create_model(
    encoder_config: BertConfig,
    tokenizer: BertTokenizer,
    criterion: MultiTaskJointLoss,
    model_type: str,
    memory_reading_module: str,
    encoder_path: str | None = None,
    entity_types: dict | None = None,
    relation_types: dict | None = None,
    prop_drop: float = 0.1,
    meta_embedding_size: int = 25,
    size_embeddings_count: int = 30,
    ed_embeddings_count: int = 300,
    token_dist_embeddings_count: int = 700,
    sentence_dist_embeddings_count: int = 50,
    mention_threshold: float = 0.5,
    coref_threshold: float = 0.5,
    rel_threshold: float = 0.5,
    position_embeddings_count: int = 700,
    mil_attention_size: int = 64,
    memory_flow_modules: list[str] | None = None,
    memory_read_modules: list[str] | None = None,
    use_entity_memory: bool = True,
    use_relation_memory: bool = True,
    memory_read_grad: bool = True,
    entity_memory_size: int | None = None,
    relation_memory_size: int | None = None,
    cache_path=None,
    coref_resolution_type: Literal['Classification', 'CosineSimilarity'] = 'Classification',
) -> MemoryRE:
    params = dict(
        config=encoder_config,
        # MemoryRE model parameters
        cls_token=tokenizer.convert_tokens_to_ids('[CLS]'),
        criterion=criterion,
        entity_types=entity_types,
        relation_types=relation_types,
        prop_drop=prop_drop,
        meta_embedding_size=meta_embedding_size,
        size_embeddings_count=size_embeddings_count,
        ed_embeddings_count=ed_embeddings_count,
        token_dist_embeddings_count=token_dist_embeddings_count,
        sentence_dist_embeddings_count=sentence_dist_embeddings_count,
        mention_threshold=mention_threshold,
        coref_threshold=coref_threshold,
        rel_threshold=rel_threshold,
        tokenizer=tokenizer,
        cache_dir=cache_path,
        # memory parameters
        memory_reading_module=memory_reading_module,
        use_entity_memory=use_entity_memory,
        use_relation_memory=use_relation_memory,
        memory_read_grad=memory_read_grad,
        memory_flow_modules=memory_flow_modules,
        memory_read_modules=memory_read_modules,
        entity_memory_size=entity_memory_size,
        relation_memory_size=relation_memory_size,
        mil_attention_size=mil_attention_size,
        coref_resolution_type=coref_resolution_type,
    )

    model_class = _MODELS[model_type]

    if encoder_path is not None:
        model = model_class.from_pretrained(encoder_path, **params)  # type: ignore
    else:
        model = model_class(**params)  # type: ignore

    # conditionally increase position embedding count
    if encoder_config.max_position_embeddings < position_embeddings_count:
        old = model.bert.embeddings.position_embeddings

        new = nn.Embedding(position_embeddings_count, encoder_config.hidden_size)
        new.weight.data[: encoder_config.max_position_embeddings, :] = old.weight.data
        model.bert.embeddings.position_embeddings = new
        model.bert.embeddings.register_buffer(
            "position_ids", torch.arange(position_embeddings_count).expand((1, -1))
        )

        model.bert.embeddings.token_type_ids = torch.zeros(
            (1, position_embeddings_count), dtype=torch.int64
        )
        encoder_config.max_position_embeddings = position_embeddings_count

    return model
