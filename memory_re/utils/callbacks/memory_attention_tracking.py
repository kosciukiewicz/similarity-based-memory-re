from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import cm
from pytorch_lightning.callbacks import Callback
from transformers import PreTrainedTokenizer

from memory_re.data.datasets.entities import EntityType, RelationType, TokenizedDocument
from memory_re.utils.logger import get_logger

LOGGER = get_logger(__name__)
matplotlib.use('Agg')


class MemoryAttentionTrackingCallback(Callback):
    def __init__(
        self,
        stage: str,
        save_dir: str | Path,
        memory_level: Literal['tokens', 'mentions'],
        memory_type: Literal['relations', 'entities'],
        entity_labels: dict[str, EntityType],
        relation_types: dict[str, RelationType],
        tokenizer: PreTrainedTokenizer,
        all_types: bool = False,
        document_ids: list[int] | None = None,
    ):
        self._stage = stage
        self._memory_level = memory_level
        self._memory_type = memory_type
        self._save_dir = Path(save_dir)
        self._entity_labels = entity_labels
        self._all_types = all_types
        self._relation_types = relation_types
        self._document_ids = document_ids
        self._tokenizer = tokenizer
        self._examples_to_store: list[int] = []
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def on_test_batch_end(  # type: ignore[override]
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: dict[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self._stage == 'test' and outputs is not None:
            doc_ids = batch['doc_ids']
            memory_modules_attentions = outputs['memory_modules_attentions']
            memory_attentions = memory_modules_attentions[f'{self._memory_level}_flow']
            entity_memory_attentions = memory_attentions['entity'].cpu().detach()
            relation_memory_attentions = memory_attentions['relation'].cpu().detach()

            docs = trainer.datamodule.test_dataset.documents  # type: ignore
            for batch_idx in range(doc_ids.shape[0]):
                doc_id = doc_ids[batch_idx].detach().cpu().item()
                valid_mentions = outputs['pos_valid_mentions'][batch_idx].detach().cpu()
                valid_mentions_masks = (
                    [outputs['pos_valid_mentions_masks']][batch_idx].detach().cpu()
                )
                valid_mentions = valid_mentions[valid_mentions_masks.bool().squeeze()]
                mention_spans = batch['mention_orig_spans'].detach().cpu()

                if self._document_ids and doc_id not in self._document_ids:
                    continue

                labels_dict: dict = (
                    self._entity_labels
                    if self._memory_type == 'entities'
                    else self._relation_types  # type: ignore
                )
                attentions = (
                    entity_memory_attentions
                    if self._memory_type == 'entities'
                    else relation_memory_attentions
                )
                labels = [label.short_name for label in labels_dict.values()]

                if self._memory_level == 'mentions':
                    self.visualize_mentions(
                        doc_id,
                        mention_spans=mention_spans[batch_idx],
                        document=docs[doc_id],
                        labels=labels,
                        valid_mentions=valid_mentions,
                        memory_attention=attentions[batch_idx],
                        save_dir=self._save_dir,
                    )
                else:
                    self.visualize_tokens(
                        doc_id,
                        document=docs[doc_id],
                        labels=labels,
                        memory_attention=attentions[batch_idx],
                        save_dir=self._save_dir,
                    )

    def visualize_mentions(
        self,
        doc_id: int,
        mention_spans: torch.Tensor,
        valid_mentions: torch.Tensor,
        document: TokenizedDocument,
        labels: list[str],
        memory_attention: torch.Tensor,
        save_dir: Path,
    ) -> None:
        valid_mention_spans = mention_spans[valid_mentions].detach().cpu()
        valid_mention_memory_attention = memory_attention[valid_mentions]
        other_mentions_mask = torch.ones(mention_spans.shape[0], dtype=torch.bool)
        other_mentions_mask[valid_mentions] = False
        other_mentions_memory_attention = memory_attention[other_mentions_mask]

        n_plots = 3 + len(labels) if self._all_types else 2
        fig, axs = plt.subplots(
            n_plots,
            2,
            figsize=(valid_mention_spans.shape[0], 12 if self._all_types else 4),
            sharey='row',
            sharex='col',
            gridspec_kw={
                'width_ratios': [4, 1],
                'height_ratios': [2] + [1] * ((2 + len(labels)) if self._all_types else 1),
            },
        )

        axs[1, 0].set_ylabel('Sum')
        color = cm.rainbow(np.linspace(0, 1, len(labels)))

        if self._all_types:
            for i, label in enumerate(labels):
                axs[n_plots - 1 - i, 0].set_ylabel(label, color=color[i])
            axs[2, 0].set_ylabel('Ratio')

        x = range(valid_mention_spans.shape[0])

        for _, _axs in enumerate(axs):
            for ax in _axs:
                ax.set_ylim(0, 1)
                ax.get_xaxis().set_visible(False)

        y_s = []
        axs[0, 0].set_xticks(x)
        axs[0, 0].axis('off')
        axs[0, 0].set_ylim(0, 0.2)

        for i in range(valid_mention_spans.shape[0]):
            mention_span = valid_mention_spans[i]
            phrase = ' '.join(
                t.phrase for t in document.tokens[mention_span[0] : mention_span[1]]  # type: ignore
            )

            mention_memory_attentions = valid_mention_memory_attention[i, :]

            axs[0, 0].text(
                i,
                0.05,
                phrase,
                rotation=90,
                fontsize=18,
                verticalalignment='bottom',
                horizontalalignment='center',
            )
            y_s.append(mention_memory_attentions)

        y = np.vstack(y_s).transpose()
        y_stacked = np.zeros_like(y[0])
        y_sum = np.sum(y, axis=0)
        axs[1, 0].bar(x, y_sum, color='b')
        axs[1, 0].set_ylim(0, np.max(y_sum) * 1.1)

        if self._all_types:
            for j in range(y.shape[0]):
                axs[n_plots - 1 - j, 0].bar(x, y[j], color=color[j])
                axs[2, 0].bar(x, y[j] / y_sum, bottom=y_stacked / y_sum, color=color[j])
                y_stacked += y[j]

        x = range(3)
        axs[0, 1].set_xticks(x)
        axs[0, 1].axis('off')

        y_s = []

        for i, (phrase, values) in enumerate(
            (
                ('max', other_mentions_memory_attention.sum(dim=1).max()),
                ('mean', other_mentions_memory_attention.sum(dim=1).mean()),
                ('min', other_mentions_memory_attention.sum(dim=1).min()),
            )
        ):
            axs[0, 1].text(
                i,
                0.05,
                f'Others ({phrase})',
                rotation=90,
                fontsize=18,
                verticalalignment='bottom',
                horizontalalignment='center',
            )

            y_s.append(values)

        y = np.vstack(y_s).transpose()
        y_sum = np.sum(y, axis=0)
        axs[1, 1].bar(x, y_sum, color='b')

        plt.tight_layout()
        plt.savefig(Path(save_dir) / f'{doc_id}.jpg', dpi=200)

    def visualize_tokens(  # noqa
        self,
        doc_id: int,
        document: TokenizedDocument,
        labels: list[str],
        memory_attention: torch.Tensor,
        save_dir: Path,
    ) -> None:
        width_ratios = [len(sentence.tokens) for sentence in document.sentences]
        n_plots = 3 + len(labels) if self._all_types else 2
        fig, axs = plt.subplots(
            n_plots,
            len(document.sentences),
            figsize=(int(0.7 * len(document.tokens)), 12 if self._all_types else 6),
            sharey='row',
            sharex='col',
            gridspec_kw={
                'width_ratios': width_ratios,
                'height_ratios': [4] + [1] * ((2 + len(labels)) if self._all_types else 1),
            },
        )

        axs[1, 0].set_ylabel('Sum')
        color = cm.rainbow(np.linspace(0, 1, len(labels)))

        if self._all_types:
            for i, label in enumerate(labels):
                axs[n_plots - 1 - i, 0].set_ylabel(label, color=color[i])
            axs[2, 0].set_ylabel('Ratio')

        for i, _axs in enumerate(axs):
            if i > 0:
                for ax in _axs:
                    ax.set_ylim(0, 1)
                    ax.get_xaxis().set_visible(False)

        y_sum_max = 0

        for i, sentence in enumerate(document.sentences):
            x = range(len(sentence.tokens))
            axs[0, i].set_xticks(x)
            axs[0, i].axis('off')
            axs[0, i].set_ylim(0, 0.2)

            y_s = []

            for j, token in enumerate(sentence.tokens):
                token_memory_attentions = memory_attention[token.span_start : token.span_end, :]
                token_memory_attentions = token_memory_attentions.sum(dim=0)

                axs[0, i].text(
                    j,
                    0.05,
                    token.phrase,
                    rotation=90,
                    fontsize=18,
                    verticalalignment='bottom',
                    horizontalalignment='center',
                )
                y_s.append(token_memory_attentions)

            y = np.vstack(y_s).transpose()
            y_stacked = np.zeros_like(y[0])
            y_sum = np.sum(y, axis=0)
            local_y_sum_max = np.max(y_sum)

            if local_y_sum_max > y_sum_max:
                y_sum_max = local_y_sum_max

            axs[1, i].bar(x, y_sum, color='b')

            if self._all_types:
                for j in range(y.shape[0]):
                    axs[n_plots - 1 - j, i].bar(x, y[j], color=color[j])
                    axs[2, i].bar(x, y[j] / y_sum, bottom=y_stacked / y_sum, color=color[j])
                    y_stacked += y[j]

        axs[1, 0].set_ylim(0, y_sum_max)

        plt.tight_layout()
        plt.savefig(Path(save_dir) / f'{doc_id}.jpg', dpi=200)
