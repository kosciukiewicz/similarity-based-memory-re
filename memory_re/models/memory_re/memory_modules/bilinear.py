import torch
import torch.nn.functional as F
from torch.nn import ModuleDict

from memory_re.models.memory_re.memory_modules.base import (
    BaseMemoryModule,
    BaseMemoryReadingModule,
    EntityClassification,
    MemoryModule,
    RelationClassification,
)


class BilinearMemoryReadingModule(BaseMemoryReadingModule):
    def __init__(
        self,
        repr_size: int,
        memory_size: int,
        memory_read_modules: list[str] | None = None,
        memory_flow_modules: list[str] | None = None,
    ):
        super().__init__()
        self._memory_read_modules = memory_read_modules or []
        self._memory_flow_modules = memory_flow_modules or []
        self._norm_memory_weights = ModuleDict(
            {
                module_name: torch.nn.Linear(
                    memory_size,
                    repr_size,
                    bias=False,
                )
                for module_name in self._memory_read_modules
            }
        )

        self._inverse_memory_weights = ModuleDict(
            {
                module_name: torch.nn.Linear(
                    memory_size,
                    repr_size,
                    bias=False,
                )
                for module_name in self._memory_flow_modules
            }
        )

    def inverse_read(
        self,
        input_reprs: torch.Tensor,
        input_mask: torch.Tensor,
        memory: torch.Tensor,
        module_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self._inverse_memory_weights[module_name].weight
        s = F.linear(memory, weights)
        attn = F.linear(input_reprs, s)
        attn[~input_mask.bool()] = -1e25
        attn = torch.nn.functional.softmax(attn, dim=1)
        mention_entity_attention = torch.sum(attn, dim=2, keepdim=True)
        return torch.mul(mention_entity_attention, input_reprs), attn

    def norm_read(
        self,
        input_reprs: torch.Tensor,
        input_mask: torch.Tensor,
        memory: torch.Tensor,
        module_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self._norm_memory_weights[module_name].weight
        attn = torch.matmul(torch.matmul(input_reprs, weights), torch.transpose(memory, 0, 1))
        attn[~input_mask.bool()] = -1e25
        attn = torch.nn.functional.softmax(attn, dim=2)
        return torch.matmul(attn, memory), attn


class MatrixMemoryModule(BaseMemoryModule):
    def __init__(
        self,
        memory_size: int,
        memory_slots: int,
    ) -> None:
        super().__init__()
        self._memory_matrix = torch.nn.Linear(memory_size, memory_slots, bias=False)

    @property
    def memory(self) -> torch.Tensor:
        return self._memory_matrix.weight


class BilinearSimilarityEntityClassification(EntityClassification):
    def __init__(self, input_size: int, memory_module: BaseMemoryModule):
        super().__init__()
        self.entity_clf_linear = torch.nn.Linear(input_size, memory_module.memory_size, False)
        self._memory_module = memory_module

    def forward(self, entity_reprs: torch.Tensor, inference: bool = True) -> torch.Tensor:
        entity_clf = self.entity_clf_linear(entity_reprs)
        return torch.nn.functional.linear(entity_clf, self._memory_module.memory)


class BilinearSimilarityRelationClassification(RelationClassification):
    def __init__(self, hidden_size: int, memory_module: MemoryModule):
        super().__init__()
        self._memory_module = memory_module
        self._relation_clf_linear = torch.nn.Linear(hidden_size, memory_module.memory_size, False)

    def forward(self, entity_pair_reprs: torch.Tensor, inference: bool = True):
        rel_clf = self._relation_clf_linear(entity_pair_reprs)
        return F.linear(rel_clf, self._memory_module.memory)
