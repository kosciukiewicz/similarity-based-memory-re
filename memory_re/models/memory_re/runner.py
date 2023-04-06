from pathlib import Path

from jerex.evaluation.joint_evaluator import JointEvaluator
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from transformers import BertConfig, BertTokenizer

from memory_re.data.datamodules.memory_re import MemoryReDataModule
from memory_re.models.memory_re import create_model
from memory_re.models.memory_re.training_wrapper import MemoryRETrainingWrapper
from memory_re.settings import PRETRAINED_MODELS_DIR
from memory_re.training.memory_re_loss import create_criterion
from memory_re.training.trainer import BaseTrainer
from memory_re.utils.callbacks import configure_callbacks
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
    encoder_config = BertConfig.from_pretrained(pretrained_model)

    data_module = MemoryReDataModule(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        train_batch_size=config.trainer.train_batch_size,
        val_batch_size=config.trainer.val_batch_size,
        num_workers=config.trainer.num_workers,
        max_span_size=config.data.max_span_size,
        neg_relation_count=config.data.neg_relation_count,
    )

    criterion = create_criterion(
        num_entity_classes=len(data_module.entity_types),
        num_relation_classes=len(data_module.relation_types),
        loss_config=config.loss,
    )

    model = create_model(
        encoder_config=encoder_config,  # type: ignore
        model_type=config.model.model_type,
        criterion=criterion,
        tokenizer=tokenizer,
        encoder_path=pretrained_model,
        entity_types=data_module.entity_types,
        relation_types=data_module.relation_types,
        prop_drop=config.model.prop_drop,
        use_entity_memory=config.model.memory.use_entity_memory,
        use_relation_memory=config.model.memory.use_relation_memory,
        memory_read_grad=config.model.memory.memory_read_grad,
        memory_flow_modules=config.model.memory.memory_flow_modules,
        memory_read_modules=config.model.memory.memory_read_modules,
        entity_memory_size=config.model.memory.entity_memory_size,
        relation_memory_size=config.model.memory.relation_memory_size,
        meta_embedding_size=config.model.meta_embedding_size,
        size_embeddings_count=config.model.size_embeddings_count,
        position_embeddings_count=config.model.position_embeddings_count,
        mention_threshold=config.model.mention_threshold,
        coref_threshold=config.model.coref_threshold,
        rel_threshold=config.model.rel_threshold,
        coref_resolution_type=config.model.coref_resolution_type,
        memory_reading_module=config.model.memory.reading_module,
    )

    evaluator = JointEvaluator(
        entity_types=data_module.entity_types,
        relation_types=data_module.relation_types,
        tokenizer=tokenizer,
    )

    training_wrapper = MemoryRETrainingWrapper(
        model=model,
        evaluator=evaluator,
        criterion=criterion,
        weight_decay=config.optimizer.weight_decay,
        learning_rate=config.optimizer.learning_rate,
        warmup_scheduler=config.optimizer.scheduler,
        warmup_proportion=config.optimizer.warmup_proportion,
        max_spans_train=config.model.max_spans_train,
        max_rel_pairs_train=config.model.max_rel_pairs_train,
        max_coref_pairs_train=config.model.max_coref_pairs_train,
        max_spans_inference=config.model.max_spans_inference,
        max_rel_pairs_inference=config.model.max_rel_pairs_inference,
        max_coref_pairs_inference=config.model.max_coref_pairs_inference,
        memory_warmup_proportion=config.model.memory.warmup_proportion,
        coref_training_samples=config.trainer.coref_training_samples,
    )

    callbacks = configure_callbacks(
        callbacks_config=config.callbacks,
        evaluator=evaluator,
        predictions_dir=Path(config.predictions_dir),
        tokenizer=tokenizer,
        entity_labels=data_module.entity_types,
        relation_types=data_module.relation_types,
    )

    trainer = BaseTrainer(
        module=training_wrapper, data_module=data_module, config=config, callbacks=callbacks
    )
    metrics = trainer.fit()

    if 'saved_pretrained_model_name' in config:
        model.save_pretrained(  # type: ignore[attr-defined]
            PRETRAINED_MODELS_DIR / config.saved_pretrained_model_name
        )

    return metrics[config.get('monitor', 'val/rel_nec/f1_micro')].item()


def test(config: DictConfig):
    pretrained_model = get_pretrained_model_name(config['pretrained_model'])
    pretrained_tokenizer = get_pretrained_model_name(config['pretrained_model'])

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_tokenizer, do_lower_case=config.data.lowercase
    )
    encoder_config = BertConfig.from_pretrained(pretrained_model)

    data_module = MemoryReDataModule(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        train_batch_size=config.trainer.train_batch_size,
        val_batch_size=config.trainer.val_batch_size,
        num_workers=config.trainer.num_workers,
        max_span_size=config.data.max_span_size,
        neg_relation_count=config.data.neg_relation_count,
    )

    criterion = create_criterion(
        num_entity_classes=len(data_module.entity_types),
        num_relation_classes=len(data_module.relation_types),
        loss_config=config.loss,
    )

    model = create_model(
        encoder_config=encoder_config,  # type: ignore
        model_type=config.model.model_type,
        tokenizer=tokenizer,
        criterion=criterion,
        encoder_path=pretrained_model,
        entity_types=data_module.entity_types,
        relation_types=data_module.relation_types,
        prop_drop=config.model.prop_drop,
        use_entity_memory=config.model.memory.use_entity_memory,
        use_relation_memory=config.model.memory.use_relation_memory,
        memory_read_grad=config.model.memory.memory_read_grad,
        memory_flow_modules=config.model.memory.memory_flow_modules,
        memory_read_modules=config.model.memory.memory_read_modules,
        entity_memory_size=config.model.memory.entity_memory_size,
        relation_memory_size=config.model.memory.relation_memory_size,
        meta_embedding_size=config.model.meta_embedding_size,
        size_embeddings_count=config.model.size_embeddings_count,
        position_embeddings_count=config.model.position_embeddings_count,
        mention_threshold=config.model.mention_threshold,
        coref_threshold=config.model.coref_threshold,
        rel_threshold=config.model.rel_threshold,
        coref_resolution_type=config.model.coref_resolution_type,
        memory_reading_module=config.model.memory.reading_module,
    )

    training_wrapper = MemoryRETrainingWrapper.load_from_checkpoint(
        config.checkpoint_path,
        model=model,
        evaluator=JointEvaluator(
            entity_types=data_module.entity_types,
            relation_types=data_module.relation_types,
            tokenizer=tokenizer,
        ),
        criterion=create_criterion(
            num_entity_classes=len(data_module.entity_types),
            num_relation_classes=len(data_module.relation_types),
            loss_config=config.loss,
        ),
        weight_decay=config.optimizer.weight_decay,
        learning_rate=config.optimizer.learning_rate,
        warmup_scheduler=config.optimizer.scheduler,
        warmup_proportion=config.optimizer.warmup_proportion,
        max_spans_train=config.model.max_spans_train,
        max_rel_pairs_train=config.model.max_rel_pairs_train,
        max_coref_pairs_train=config.model.max_coref_pairs_train,
        max_spans_inference=config.model.max_spans_inference,
        max_rel_pairs_inference=config.model.max_rel_pairs_inference,
        max_coref_pairs_inference=config.model.max_coref_pairs_inference,
        memory_warmup_proportion=config.model.memory.warmup_proportion,
        coref_training_samples=config.trainer.coref_training_samples,
    )

    evaluator = JointEvaluator(
        entity_types=data_module.entity_types,
        relation_types=data_module.relation_types,
        tokenizer=tokenizer,
    )

    callbacks = configure_callbacks(
        callbacks_config=config.callbacks,
        evaluator=evaluator,
        tokenizer=tokenizer,
        entity_labels=data_module.entity_types,
        relation_types=data_module.relation_types,
        predictions_dir=Path(config.predictions_dir),
    )

    trainer = Trainer(
        move_metrics_to_cpu=True,
        gpus=config.trainer.gpus,
        callbacks=callbacks,
    )

    metrics = trainer.test(training_wrapper, datamodule=data_module)
    LOGGER.info(metrics[0])
    return metrics[0][config.get('monitor', 'test/rel_nec/f1_micro')]
