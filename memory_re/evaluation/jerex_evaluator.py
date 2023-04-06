import os
import sys
from copy import deepcopy
from typing import Any, Tuple

from jerex.evaluation.evaluator import Evaluator
from torch import Tensor

from memory_re.data.datasets.entities import TokenizedDocument


class JEREXEvaluator:
    def __init__(
        self,
        jerex_evaluator: Evaluator,
        prefix: str = '',
    ):
        super().__init__()
        self._ground_truth = None
        self._predictions: list[Tuple] = []
        self._prefix = prefix
        self._jerex_evaluator = jerex_evaluator

    def convert_batch(self, *args, **kwargs):
        return self._jerex_evaluator.convert_batch(*args, **kwargs)

    def set_gt_documents(self, documents: list[TokenizedDocument]):
        self._ground_truth = self._jerex_evaluator.convert_gt(documents)  # type: ignore

    def update(
        self,
        model_outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> None:
        predictions = self._jerex_evaluator.convert_batch(**model_outputs, batch=batch)
        self._predictions.extend(predictions)

    def compute(self) -> dict[str, Any]:
        sys.stdout = open(os.devnull, 'w')
        metrics = self._jerex_evaluator.compute_metrics(
            self._ground_truth[: len(self._predictions)], self._predictions  # type: ignore
        )
        sys.stdout = sys.__stdout__

        return {
            f"{self._prefix}{metrics_key}/f1_micro": metrics_values['f1_micro']
            for metrics_key, metrics_values in metrics.items()
        }

    def reset(self) -> None:
        self._predictions = []

    def clone(self, prefix: str = '') -> 'JEREXEvaluator':
        evaluator = deepcopy(self)
        evaluator._prefix = prefix
        return evaluator
