import os
from pathlib import Path
from typing import Any

import jinja2
import wandb
from jerex.evaluation.joint_evaluator import JointEvaluator
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

from memory_re.data.datamodules.jerex import JEREXDataModule
from memory_re.data.datamodules.memory_re import MemoryReDataModule
from memory_re.settings import STORAGE_DIR
from memory_re.utils.logger import get_logger
from memory_re.visualization.relation_extraction.predictions.to_html import convert_example

LOGGER = get_logger('PredictionVisualisationCallback')
TEMPLATE_PATH = STORAGE_DIR / 'misc' / 'relation_extraction_example_template.html'


class RelationExtractionPredictionsHtmlSaverCallback(Callback):
    def __init__(
        self,
        jerex_evaluator: JointEvaluator,
        destination_dir: str | Path,
        stage: str = 'test',
        log_predictions: bool = False,
        document_ids: list[int] | None = None,
    ):
        if isinstance(destination_dir, str):
            destination_dir = Path(destination_dir)

        self.destination_filepath = destination_dir
        self._stage = stage
        self._evaluator = jerex_evaluator
        self._examples_to_store: list[dict] = []
        self._log_predictions = log_predictions
        self._documents_ids = document_ids

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._stage == 'val':
            self._init_examples()

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._stage == 'test':
            self._init_examples()

    def _init_examples(self) -> None:
        if self.destination_filepath.exists():
            os.remove(self.destination_filepath)
        else:
            self.destination_filepath.parent.mkdir(exist_ok=True, parents=True)

        self._examples_to_store = []

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any] | None,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self._stage == 'val':
            if outputs is not None and 'val_predictions' in outputs:
                self._get_visualisation(trainer, outputs['val_predictions'])
            else:
                LOGGER.warning('No predictions to save in %s output', self._stage)

    def on_test_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any] | None,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self._stage == 'test':
            if outputs is not None and 'test_predictions' in outputs:
                self._get_visualisation(trainer, outputs['test_predictions'])
            else:
                LOGGER.warning('No predictions to save in %s output', self._stage)

    def _get_visualisation(self, trainer: Trainer, predictions: list[dict[str, Any]]):
        if not (
            isinstance(trainer.datamodule, JEREXDataModule)  # type: ignore[attr-defined]
            or isinstance(trainer.datamodule, MemoryReDataModule)  # type: ignore[attr-defined]
        ):
            raise ValueError(
                'Only JEREXDataModule and MemoryReDataModule is supported for using with this callback'
            )

        if self._stage == 'test':
            docs = trainer.datamodule.test_dataset.documents  # type: ignore
        elif self._stage == 'val':
            docs = trainer.datamodule.valid_dataset.documents  # type: ignore
        else:
            raise ValueError()

        for prediction in predictions:
            doc_id = prediction['doc_id']
            if self._documents_ids is None or doc_id in self._documents_ids:
                (
                    pred_mentions,
                    pred_clusters,
                    pred_entities,
                    pred_relations,
                    pred_relations_et,
                ) = prediction['doc_predictions']
                ground_truth = self._evaluator.convert_gt([docs[doc_id]])[0]
                gt_mentions, gt_clusters, gt_entities, gt_relations, gt_relations_et = ground_truth
                example_doc = convert_example(
                    doc=docs[doc_id],
                    gt_mentions=gt_mentions,
                    pred_mentions=pred_mentions,
                    gt_clusters=gt_clusters,
                    pred_clusters=pred_clusters,
                    gt_entities=gt_entities,
                    pred_entities=pred_entities,
                    gt_relations=gt_relations_et,
                    pred_relations=pred_relations_et,
                )
                self._examples_to_store.append(example_doc)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._stage == 'val':
            if self._examples_to_store is not None:
                self._store_examples(self._examples_to_store)

            if self._log_predictions:
                self._log_predictions_file(trainer)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._stage == 'test':
            if self._examples_to_store is not None:
                self._store_examples(self._examples_to_store)

            if self._log_predictions:
                self._log_predictions_file(trainer)

    def _log_predictions_file(self, trainer: Trainer) -> None:
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                wandb.log(
                    {
                        'step': trainer.global_step,
                        f'{self._stage}-documents': wandb.Html(open(self.destination_filepath)),
                    }
                )

    def _store_examples(self, examples: list[dict]):
        # read template
        with open(TEMPLATE_PATH) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(docs=examples).dump(str(self.destination_filepath))
