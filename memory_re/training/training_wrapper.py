import abc

import pytorch_lightning as pl
import torch
import transformers
from torch import nn

from memory_re.training.optim.schedulers import get_noam_warmup_scheduler


class BaseTrainingWrapper(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        warmup_proportion: float | None = None,
        scheduler: str = 'WarmupLinear',
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self._model = model
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate
        self._warmup_proportion = warmup_proportion
        self._scheduler = scheduler

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer is None:
            raise ValueError('BaseTrainer not inizialized')

        if self.trainer.max_steps is not None and self.trainer.max_steps != -1:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())  # type: ignore[attr-defined]
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_devices)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer | tuple[list, list]:
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters = list(self._model.named_parameters())
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
                'weight_decay': self._weight_decay,
            },
            {
                'params': [p for n, p in parameters if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        optimizer: torch.optim.Optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=self._learning_rate, correct_bias=False  # type: ignore
        )
        optimizers = [optimizer]

        if self._warmup_proportion is not None:
            scheduler = self._get_scheduler(optimizer, self._warmup_proportion)
            schedulers = [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]
        else:
            schedulers = []

        return optimizers, schedulers

    def _get_scheduler(self, optimizer: torch.optim.Optimizer, warmup_proportion: int | float):
        warmup_steps = self._get_warmup_steps(warmup_proportion)
        scheduler = self._scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=self.num_training_steps
            )
        elif scheduler == 'onecyclelr':
            return torch.optim.lr_scheduler.OneCycleLR(  # type: ignore[attr-defined]
                optimizer, max_lr=self._learning_rate, total_steps=self.num_training_steps
            )
        elif scheduler == 'noam':
            return get_noam_warmup_scheduler(optimizer, num_warmup_steps=warmup_steps)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _get_warmup_steps(self, warmup_proportion: int | float) -> int:
        if isinstance(warmup_proportion, int):
            return warmup_proportion
        else:
            return int(self.num_training_steps * warmup_proportion)
