from typing import Any

from jerex.evaluation.joint_evaluator import JointEvaluator
from omegaconf import DictConfig
from pytorch_lightning import Callback

from memory_re.utils.callbacks.memory_attention_tracking import MemoryAttentionTrackingCallback
from memory_re.utils.callbacks.relation_extraction_prediction_html_saver import (
    RelationExtractionPredictionsHtmlSaverCallback,
)


def configure_callbacks(callbacks_config: list[DictConfig], **kwargs: Any) -> list[Callback]:
    return [_configure_callback(callback_config, **kwargs) for callback_config in callbacks_config]


def _configure_callback(callback_config: DictConfig, **kwargs: Any) -> Callback:
    match callback_config.callback_type:
        case 'RelationExtractionPredictionSaver':
            return _configure_relation_extraction_prediction_saver(callback_config, **kwargs)
        case 'MemoryAttentionTrackingCallback':
            return MemoryAttentionTrackingCallback(
                stage=callback_config.stage,
                save_dir=callback_config.save_dir,
                all_types=callback_config.all_types,
                memory_type=callback_config.memory_type,
                memory_level=callback_config.memory_level,
                tokenizer=kwargs['tokenizer'],
                entity_labels=kwargs['entity_labels'],
                relation_types=kwargs['relation_types'],
                document_ids=callback_config.get('document_ids'),
            )
        case _:
            raise ValueError(f'Callback not supported {callback_config.callback_type}')


def _configure_relation_extraction_prediction_saver(
    config: DictConfig, evaluator: JointEvaluator, **kwargs: Any
) -> Callback:
    return RelationExtractionPredictionsHtmlSaverCallback(
        jerex_evaluator=evaluator,
        destination_dir=config.destination_filepath,
        document_ids=config.get('document_ids'),
        log_predictions=config.get('log_predictions', False),
        stage=config.stage,
    )
