from pathlib import Path

import jerex.models as jerex_models
import torch
from jerex.evaluation.joint_evaluator import JointEvaluator
from jerex.loss import JointLoss
from torch import nn
from transformers import BertConfig, BertPreTrainedModel, BertTokenizer


def _create_model(
    model_class: BertPreTrainedModel,
    encoder_config: BertConfig,
    tokenizer: BertTokenizer,
    entity_types: dict,
    relation_types: dict,
    encoder_path=None,
    prop_drop: float = 0.1,
    meta_embedding_size: int = 25,
    size_embeddings_count: int = 10,
    ed_embeddings_count: int = 300,
    token_dist_embeddings_count: int = 700,
    sentence_dist_embeddings_count: int = 50,
    mention_threshold: float = 0.5,
    coref_threshold: float = 0.5,
    rel_threshold: float = 0.5,
    position_embeddings_count: int = 700,
    cache_path=None,
):
    params = dict(
        config=encoder_config,
        # JEREX model parameters
        cls_token=tokenizer.convert_tokens_to_ids('[CLS]'),
        entity_types=len(entity_types),
        relation_types=len(relation_types),
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
    )

    if encoder_path is not None:
        model = model_class.from_pretrained(encoder_path, **params)  # type: ignore[attr-defined]
    else:
        model = model_class(**params)  # type: ignore[operator]

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


class JEREXModelWrapper(nn.Module):
    def __init__(
        self,
        model_type: str,
        tokenizer_path: str,
        encoder_path: str,
        entity_types: dict,
        relation_types: dict,
        lowercase: bool = False,
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
    ):
        super().__init__()
        self.model_class = jerex_models.get_model(model_type)
        self._tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=lowercase)

        self._encoder_config = BertConfig.from_pretrained(encoder_path)
        self._entity_types = entity_types
        self._relation_types = relation_types

        self.model = _create_model(
            self.model_class,
            encoder_config=self._encoder_config,  # type: ignore[arg-type]
            tokenizer=self._tokenizer,
            encoder_path=encoder_path,
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
            position_embeddings_count=position_embeddings_count,
        )

    def get_evaluator(self) -> JointEvaluator:
        return self.model_class.EVALUATOR(self._entity_types, self._relation_types, self._tokenizer)

    def get_loss(self, task_weights: list[float]) -> JointLoss:
        return self.model.LOSS(task_weights)

    def forward(self, *args, **batch):
        return self.model(
            *args,
            **batch,
        )

    def save(self, destination_dir: str | Path) -> None:
        pass
