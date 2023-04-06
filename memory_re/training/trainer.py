import argparse
from typing import Dict

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import GPUStatsMonitor, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, LightningLoggerBase, WandbLogger

from memory_re.utils.callbacks.best_metric_value_tracking import BestMetricValueTrackingCallback
from memory_re.utils.logger import get_logger

LOGGER = get_logger(name='Trainer')


class BaseTrainer:
    def __init__(
        self,
        config: DictConfig,
        data_module: pl.LightningDataModule,
        module: pl.LightningModule,
        callbacks: list[Callback] | None = None,
    ):
        super().__init__()
        self._config = config
        self._module = module
        self._data_module = data_module
        self._additional_callbacks = callbacks or []
        self._configure_callbacks()

    def fit(self) -> Dict:
        loggers = self._get_loggers()
        for logger in loggers:
            logger.log_hyperparams(argparse.Namespace(**self._config))

        trainer = Trainer(
            logger=loggers,
            callbacks=self._get_training_callbacks(),
            move_metrics_to_cpu=True,
            enable_checkpointing=self._config.get('saver') is not None,
            gpus=self._config.trainer.gpus,
            max_epochs=self._config.trainer.max_epochs,
            log_every_n_steps=self._config.trainer.log_every_n_steps,
            num_sanity_val_steps=0,
            resume_from_checkpoint=self._config.get('checkpoint_path'),
            gradient_clip_val=self._config.trainer.get('gradient_clip_val'),
        )

        trainer.fit(
            model=self._module,
            datamodule=self._data_module,
        )

        final_results = self._get_final_results(trainer)
        self.finish(loggers=loggers)
        return final_results

    def _get_final_results(self, trainer: pl.Trainer) -> dict[str, float]:
        final_results = trainer.callback_metrics

        if self._best_metric_value_tracking_callback:
            final_results.update(self._best_metric_value_tracking_callback.get_metrics())

        if self._config.trainer.final_test_evaluation:
            self._data_module.setup('test')
            test_metrics = trainer.test(
                self._module, ckpt_path='best', datamodule=self._data_module
            )[0]
            final_results.update(test_metrics)

        return final_results

    @staticmethod
    def finish(loggers: list[pl.loggers.LightningLoggerBase]) -> None:
        """Makes sure everything closed properly."""

        for logger in loggers:
            if isinstance(logger, pl.loggers.wandb.WandbLogger):
                wandb.finish()

    def _get_training_callbacks(self) -> list[Callback]:
        callbacks = [
            *self._additional_callbacks,
            LearningRateMonitor(logging_interval='step'),
            GPUStatsMonitor(),
        ]

        if self._best_metric_value_tracking_callback:
            callbacks.append(self._best_metric_value_tracking_callback)

        if self._checkpoint_callback:
            callbacks.append(self._checkpoint_callback)

        return callbacks

    def _get_loggers(self) -> list[LightningLoggerBase]:
        loggers: list[LightningLoggerBase] = []

        if 'loggers' in self._config:
            loggers.extend(self._configure_loggers(self._config.loggers))

        return loggers

    def _configure_loggers(self, logger_config: DictConfig) -> list[LightningLoggerBase]:
        loggers: list[LightningLoggerBase] = []
        if 'wandb' in logger_config:
            LOGGER.info(str(logger_config.wandb.save_dir))
            loggers.append(
                WandbLogger(
                    project=logger_config.wandb.project,
                    save_dir=logger_config.wandb.save_dir,
                )
            )
        if 'csv' in logger_config:
            LOGGER.info(str(logger_config.csv.save_dir))
            loggers.append(
                CSVLogger(
                    name=logger_config.csv.name,
                    version=logger_config.csv.version,
                    save_dir=logger_config.csv.save_dir,
                    flush_logs_every_n_steps=self._config.trainer.flush_logs_every_n_steps,
                )
            )

        return loggers

    def _configure_callbacks(self) -> None:
        self._checkpoint_callback: ModelCheckpoint | None = None
        self._best_metric_value_tracking_callback: BestMetricValueTrackingCallback | None = None

        if self._config.get('saver') is not None:
            self._checkpoint_callback = self._configure_checkpoint_callback(self._config.saver)

        if 'best_metric_monitor' in self._config:
            self._best_metric_value_tracking_callback = self._configure_best_metric_callback(
                self._config.best_metric_monitor
            )

    @staticmethod
    def _configure_best_metric_callback(config: DictConfig) -> BestMetricValueTrackingCallback:
        return BestMetricValueTrackingCallback(
            monitor=config.name,
            logging_prefix=config.logging_prefix,
            additional_metrics=config.additional_metrics,
        )

    @staticmethod
    def _configure_checkpoint_callback(config: DictConfig) -> ModelCheckpoint:
        return ModelCheckpoint(
            monitor=config.monitor,
            filename=config.filename,
            save_top_k=config.save_top_k,
            mode=config.mode,
            save_last=config.save_last,
            dirpath=config.checkpoints_dir,
            auto_insert_metric_name=False,
        )
