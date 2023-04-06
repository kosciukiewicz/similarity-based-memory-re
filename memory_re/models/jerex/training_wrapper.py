from torch import Tensor

from memory_re.evaluation.jerex_evaluator import JEREXEvaluator
from memory_re.models.jerex.model_wrapper import JEREXModelWrapper
from memory_re.training.training_wrapper import BaseTrainingWrapper


class JEREXTrainingWrapper(BaseTrainingWrapper):
    def __init__(
        self,
        model: JEREXModelWrapper,
        mention_weight: float = 1.0,
        entity_weight: float = 1.0,
        coref_weight: float = 1.0,
        relation_weight: float = 1.0,
        max_spans_train: int | None = None,
        max_spans_inference: int | None = None,
        max_coref_pairs_train: int | None = None,
        max_coref_pairs_inference: int | None = None,
        max_rel_pairs_train: int | None = None,
        max_rel_pairs_inference: int | None = None,
        learning_rate: float = 5e-5,
        warmup_proportion: float = 0.1,
        weight_decay: float = 0.01,
        warmup_scheduler: str = 'WarmupLinear',
    ):
        super().__init__(
            model,
            learning_rate=learning_rate,
            warmup_proportion=warmup_proportion,
            weight_decay=weight_decay,
            scheduler=warmup_scheduler,
        )
        self._model = model
        task_weights = [mention_weight, coref_weight, entity_weight, relation_weight]
        self._criterion = self._model.get_loss(task_weights=task_weights)
        self.evaluator = self._model.get_evaluator()
        self._val_evaluator = JEREXEvaluator(self.evaluator, prefix='val/')
        self._test_evaluator = JEREXEvaluator(self.evaluator, prefix='test/')

        self._max_spans_train = max_spans_train
        self._max_spans_inference = max_spans_inference
        self._max_coref_pairs_train = max_coref_pairs_train
        self._max_coref_pairs_inference = max_coref_pairs_inference
        self._max_rel_pairs_train = max_rel_pairs_train
        self._max_rel_pairs_inference = max_rel_pairs_inference

    def setup(self, stage: str | None = None) -> None:
        if stage == 'fit':
            self._val_evaluator.set_gt_documents(
                self.trainer.datamodule.valid_dataset.documents  # type: ignore
            )
        if stage == 'test':
            self._test_evaluator.set_gt_documents(
                self.trainer.datamodule.test_dataset.documents  # type: ignore
            )

    def model_forward(self, inference=False, **batch) -> dict[str, Tensor]:
        max_spans = self._max_spans_train if not inference else self._max_spans_inference
        max_coref_pairs = (
            self._max_coref_pairs_train if not inference else self._max_coref_pairs_inference
        )
        max_rel_pairs = (
            self._max_rel_pairs_train if not inference else self._max_rel_pairs_inference
        )

        outputs = self._model(
            **batch,
            max_spans=max_spans,
            max_coref_pairs=max_coref_pairs,
            max_rel_pairs=max_rel_pairs,
            inference=inference,
        )

        return outputs

    def training_step(self, batch, batch_idx):
        """Implements a training step, i.e. calling of forward pass and loss computation"""
        # this method is called by PL for every training step
        # the returned loss is optimized
        outputs = self.model_forward(inference=False, **batch)
        losses = self._criterion.compute(**outputs, **batch)
        loss = losses['loss']

        for tag, value in losses.items():
            self.log('train/%s' % tag, value.item(), prog_bar=True, logger=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        output = self.model_forward(inference=True, **batch)
        self._val_evaluator.update(output, batch=batch)

        predictions = self.evaluator.convert_batch(**output, batch=batch)
        return {
            'doc_ids': batch['doc_ids'],
            'val_predictions': [
                {'doc_id': doc_id.item(), 'doc_predictions': doc_predictions}
                for doc_id, doc_predictions in zip(batch['doc_ids'], predictions)
            ],
        }

    def validation_epoch_end(self, outputs: Tensor) -> None:  # type: ignore
        self.log_dict(
            self._val_evaluator.compute(), prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self._val_evaluator.reset()

    def test_step(self, batch, batch_idx):
        output = self.model_forward(inference=True, **batch)
        self._test_evaluator.update(output, batch=batch)

        predictions = self.evaluator.convert_batch(**output, batch=batch)
        return {
            'doc_ids': batch['doc_ids'],
            'test_predictions': [
                {'doc_id': doc_id.item(), 'doc_predictions': doc_predictions}
                for doc_id, doc_predictions in zip(batch['doc_ids'], predictions)
            ],
        }

    def test_epoch_end(self, outputs: Tensor) -> None:  # type: ignore
        self.log_dict(
            self._test_evaluator.compute(), prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        self._test_evaluator.reset()
