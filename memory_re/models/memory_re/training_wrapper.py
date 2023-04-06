from typing import Literal

from jerex.evaluation.joint_evaluator import JointEvaluator
from torch import Tensor

from memory_re.evaluation.jerex_evaluator import JEREXEvaluator
from memory_re.models.memory_re.memory_re import MemoryRE
from memory_re.training.memory_re_loss import MultiTaskJointLoss
from memory_re.training.training_wrapper import BaseTrainingWrapper


class MemoryRETrainingWrapper(BaseTrainingWrapper):
    def __init__(
        self,
        model: MemoryRE,
        evaluator: JointEvaluator,
        criterion: MultiTaskJointLoss | None = None,
        max_spans_train: int | None = None,
        max_spans_inference: int | None = None,
        max_coref_pairs_train: int | None = None,
        max_coref_pairs_inference: int | None = None,
        max_rel_pairs_train: int | None = None,
        max_rel_pairs_inference: int | None = None,
        memory_warmup_proportion: float = 0.1,
        coref_training_samples: Literal['Pairs', 'AdjMatrix', 'Triples'] = 'Pairs',
        learning_rate: float = 5e-5,
        warmup_proportion: float = 0.1,
        weight_decay: float = 0.01,
        warmup_scheduler: str = 'WarmupLinear',
    ):
        super().__init__(
            model,  # type: ignore
            learning_rate=learning_rate,
            warmup_proportion=warmup_proportion,
            weight_decay=weight_decay,
            scheduler=warmup_scheduler,
        )
        self._model = model  # type: ignore
        self._criterion = criterion
        self._evaluator = evaluator
        self._val_evaluator = JEREXEvaluator(evaluator, prefix='val/')
        self._test_evaluator = JEREXEvaluator(evaluator, prefix='test/')

        self._max_spans_train = max_spans_train
        self._max_spans_inference = max_spans_inference
        self._max_coref_pairs_train = max_coref_pairs_train
        self._max_coref_pairs_inference = max_coref_pairs_inference
        self._max_rel_pairs_train = max_rel_pairs_train
        self._max_rel_pairs_inference = max_rel_pairs_inference

        self._memory_warmup_proportion = memory_warmup_proportion
        self._memory_warmup_steps = 0

        self._coref_training_samples = coref_training_samples

    def setup(self, stage: str | None = None) -> None:
        if stage == 'fit':
            self._memory_warmup_steps = self._get_warmup_steps(self._memory_warmup_proportion)
            self._val_evaluator.set_gt_documents(
                self.trainer.datamodule.valid_dataset.documents  # type: ignore
            )
        if stage == 'test':
            self._test_evaluator.set_gt_documents(
                self.trainer.datamodule.test_dataset.documents  # type: ignore
            )

    @property
    def use_memory(self) -> bool:
        is_memory_warmup = self.global_step < self._memory_warmup_steps
        return not is_memory_warmup

    def get_max_spans(self, inference: bool = False) -> int | None:
        return self._max_spans_train if not inference else self._max_spans_inference

    def forward_model(self, inference=False, **batch) -> dict[str, Tensor]:
        max_coref_pairs = (
            self._max_coref_pairs_train if not inference else self._max_coref_pairs_inference
        )
        max_rel_pairs = (
            self._max_rel_pairs_train if not inference else self._max_rel_pairs_inference
        )

        outputs = self._model(
            **batch,
            max_spans=self.get_max_spans(inference=inference),
            max_coref_pairs=max_coref_pairs,
            max_rel_pairs=max_rel_pairs,
            use_memory=self.use_memory,
            inference=inference,
            criterion=self._criterion,
        )

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward_model(inference=False, **batch)

        losses = self._criterion.compute(
            **outputs,
            **batch,
            batch_id=batch_idx,
            training_step=self.global_step,
            num_training_steps=self.num_training_steps,
        )

        for tag, value in losses.items():
            self.log('train/%s' % tag, value.item(), prog_bar=True, logger=True)

        return {
            'loss': losses['loss'],
        }

    def validation_step(self, batch, batch_idx):
        output = self.forward_model(inference=True, **batch)
        memory_modules_attentions = output.pop('memory_modules_attentions')
        valid_mentions = output.pop('valid_mentions')
        valid_mention_sample_masks = output.pop('valid_mention_sample_masks')
        self._val_evaluator.update(output, batch=batch)

        predictions = self._evaluator.convert_batch(**output, batch=batch)
        return {
            'doc_ids': batch['doc_ids'],
            'pos_valid_mentions': valid_mentions,
            'pos_valid_mentions_masks': valid_mention_sample_masks,
            'memory_modules_attentions': memory_modules_attentions,
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
        output = self.forward_model(inference=True, **batch)
        memory_modules_attentions = output.pop('memory_modules_attentions')
        valid_mentions = output.pop('valid_mentions')
        valid_mention_sample_masks = output.pop('valid_mention_sample_masks')
        self._test_evaluator.update(output, batch=batch)

        predictions = self._evaluator.convert_batch(**output, batch=batch)
        return {
            'doc_ids': batch['doc_ids'],
            'pos_valid_mentions': valid_mentions,
            'pos_valid_mentions_masks': valid_mention_sample_masks,
            'memory_modules_attentions': memory_modules_attentions,
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
