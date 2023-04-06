from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer, seed_everything
from transformers import BertTokenizer

from memory_re.data.datamodules.jerex import JEREXDataModule
from memory_re.models.jerex.model_wrapper import JEREXModelWrapper
from memory_re.models.jerex.training_wrapper import JEREXTrainingWrapper
from memory_re.settings import PRETRAINED_MODELS_DIR
from memory_re.training.trainer import BaseTrainer
from memory_re.utils.callbacks import configure_callbacks
from memory_re.utils.callbacks.relation_extraction_prediction_html_saver import (
    RelationExtractionPredictionsHtmlSaverCallback,
)
from memory_re.utils.logger import get_logger
from memory_re.utils.pretrained_models import get_pretrained_model_name

LOGGER = get_logger(__name__)


def train(config: DictConfig) -> float:
    if 'seed' in config:
        LOGGER.info(f"Setting random seed value to {config.seed}")
        seed_everything(config.seed)

    pretrained_model = get_pretrained_model_name(config['pretrained_model'])
    pretrained_tokenizer = get_pretrained_model_name(config['pretrained_model'])

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_tokenizer, do_lower_case=config.data.lowercase
    )

    data_module = JEREXDataModule(
        dataset_name=config['dataset_name'],
        tokenizer=tokenizer,
        train_batch_size=config['trainer']['train_batch_size'],
        val_batch_size=config['trainer']['val_batch_size'],
        num_workers=config['trainer']['num_workers'],
        task_type=config['data']['task_type'],
        max_span_size=config['data']['max_span_size'],
    )

    model = JEREXModelWrapper(
        encoder_path=pretrained_model,
        tokenizer_path=pretrained_tokenizer,
        entity_types=data_module.entity_types,
        relation_types=data_module.relation_types,
        model_type=config['model']['model_type'],
        lowercase=config['data']['lowercase'],
        prop_drop=config['model']['prop_drop'],
        meta_embedding_size=config['model']['meta_embedding_size'],
        size_embeddings_count=config['model']['size_embeddings_count'],
        position_embeddings_count=config['model']['position_embeddings_count'],
        mention_threshold=config['model']['mention_threshold'],
        coref_threshold=config['model']['coref_threshold'],
        rel_threshold=config['model']['rel_threshold'],
    )

    training_wrapper = JEREXTrainingWrapper(
        model=model,
        entity_weight=config['loss']['entity_weight'],
        relation_weight=config['loss']['relation_weight'],
        coref_weight=config['loss']['coref_weight'],
        mention_weight=config['loss']['mention_weight'],
        max_spans_train=config['model']['max_spans_train'],
        max_rel_pairs_train=config['model']['max_rel_pairs_train'],
        max_coref_pairs_train=config['model']['max_coref_pairs_train'],
        max_spans_inference=config['model']['max_spans_inference'],
        max_rel_pairs_inference=config['model']['max_rel_pairs_inference'],
        max_coref_pairs_inference=config['model']['max_coref_pairs_inference'],
        weight_decay=config['optimizer']['weight_decay'],
        learning_rate=config['optimizer']['learning_rate'],
        warmup_proportion=config['optimizer']['warmup_proportion'],
    )

    callbacks = configure_callbacks(
        callbacks_config=config.callbacks, evaluator=training_wrapper.evaluator
    )
    trainer = BaseTrainer(
        module=training_wrapper, data_module=data_module, config=config, callbacks=callbacks
    )
    metrics = trainer.fit()

    if 'saved_pretrained_model_name' in config:
        model.model.save_pretrained(PRETRAINED_MODELS_DIR / config.saved_pretrained_model_name)

    return metrics[config.get('monitor', 'val/rel_nec/f1_micro')].item()


def test(config: DictConfig) -> None:
    pretrained_model = get_pretrained_model_name(config['pretrained_model'])
    pretrained_tokenizer = get_pretrained_model_name(config['pretrained_model'])

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_tokenizer, do_lower_case=config.data.lowercase
    )

    data_module = JEREXDataModule(
        dataset_name=config['dataset_name'],
        tokenizer=tokenizer,
        train_batch_size=config['trainer']['train_batch_size'],
        val_batch_size=config['trainer']['val_batch_size'],
        num_workers=config['trainer']['num_workers'],
        task_type=config['data']['task_type'],
        max_span_size=config['data']['max_span_size'],
    )

    model = JEREXModelWrapper(
        encoder_path=pretrained_model,
        tokenizer_path=pretrained_tokenizer,
        entity_types=data_module.entity_types,
        relation_types=data_module.relation_types,
        model_type=config['model']['model_type'],
        meta_embedding_size=config['model']['meta_embedding_size'],
        size_embeddings_count=config['model']['size_embeddings_count'],
        position_embeddings_count=config['model']['position_embeddings_count'],
        mention_threshold=config['model']['mention_threshold'],
        coref_threshold=config['model']['coref_threshold'],
        rel_threshold=config['model']['rel_threshold'],
    )

    training_wrapper = JEREXTrainingWrapper.load_from_checkpoint(
        config.checkpoint_path,
        model=model,
        max_spans_inference=config['model']['max_spans_inference'],
        max_rel_pairs_inference=config['model']['max_rel_pairs_inference'],
        max_coref_pairs_inference=config['model']['max_coref_pairs_inference'],
    )

    callbacks: list[Callback] = []

    if 'prediction_saver' in config:
        callbacks.append(
            RelationExtractionPredictionsHtmlSaverCallback(
                jerex_evaluator=model.get_evaluator(),
                destination_dir=config.prediction_saver.destination_filepath,
                stage='test',
            )
        )

    trainer = Trainer(
        callbacks=callbacks,
        move_metrics_to_cpu=True,
        gpus=config.trainer.gpus,
    )

    trainer.test(training_wrapper, datamodule=data_module)
