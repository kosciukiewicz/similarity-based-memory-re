import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from memory_re.utils.logger import get_logger

LOGGER = get_logger(__name__)


class BestMetricValueTrackingCallback(Callback):
    def __init__(
        self,
        monitor: str,
        mode: str = 'max',
        logging_prefix: str = 'best/',
        additional_metrics: list[str] | None = None,
    ):
        self._monitor = monitor
        value_mode: tuple[torch.Tensor, str] = self._init_monitor_mode(mode)
        self.best_value: float = value_mode[0].item()
        self.mode: str = value_mode[1]
        self.additional_metrics_best: dict[str, float] = {}
        self.prefix = logging_prefix
        self.additional_metrics = additional_metrics

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics

        if self._monitor in metrics:
            monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
            current = metrics[self._monitor]

            if monitor_op(current, self.best_value):
                self.update_best(metrics)
        else:
            m = (
                f"BestMetricValueTrackingCallback(monitor='{self._monitor}') not found in the "
                f"returned metrics: {list(metrics.keys())}. "
                f"HINT: Did you call self.log('{self._monitor}', value) in the LightningModule?"
            )
            LOGGER.warning(m)

    def update_best(self, metrics: dict[str, float]) -> None:
        self.best_value = metrics[self._monitor]
        self.additional_metrics_best = self._get_additional_metrics(metrics)

    def _get_additional_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if self.additional_metrics is not None:
            return {
                metric_name: metric_value
                for metric_name, metric_value in metrics.items()
                if metric_name in self.additional_metrics
            }
        else:
            return {}

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if logger := trainer.logger:
            logger.log_metrics(self.get_metrics(), step=trainer.global_step)

    def get_metrics(self) -> dict[str, float]:
        metric_to_return = {
            (self.prefix + metric_name): metric_value
            for metric_name, metric_value in self.additional_metrics_best.items()
        }
        metric_to_return[self.prefix + self._monitor] = self.best_value
        return metric_to_return

    @staticmethod
    def _init_monitor_mode(mode: str) -> tuple[torch.Tensor, str]:
        torch_inf = torch.tensor(np.Inf)
        mode_dict = {"min": (torch_inf, "min"), "max": (-torch_inf, "max")}

        if mode not in mode_dict:
            raise MisconfigurationException(
                f"`mode` can be {', '.join(mode_dict.keys())} but got {mode}"
            )

        return mode_dict[mode]
