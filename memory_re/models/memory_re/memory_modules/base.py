from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseMemoryReadingModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def inverse_read(
        self,
        input_reprs: torch.Tensor,
        input_mask: torch.Tensor,
        memory: torch.Tensor,
        module_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def norm_read(
        self,
        input_reprs: torch.Tensor,
        input_mask: torch.Tensor,
        memory: torch.Tensor,
        module_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class BaseMemoryModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def memory(self) -> torch.Tensor:
        pass

    @property
    def memory_size(self) -> int:
        return self.memory.shape[1]


class EntityClassification(nn.Module, ABC):
    @abstractmethod
    def forward(self, entity_reprs: torch.Tensor, inference: bool = True) -> torch.Tensor:
        pass


class RelationClassification(nn.Module, ABC):
    @abstractmethod
    def forward(self, entity_pair_reprs: torch.Tensor, inference: bool = True):
        pass


class MemoryModule(nn.Module, ABC):
    def __init__(
        self,
        reading_module: BaseMemoryReadingModule,
        memory_module: BaseMemoryModule,
    ):
        super().__init__()
        self.reading_module = reading_module
        self.memory_module = memory_module

    @property
    def memory(self) -> torch.Tensor:
        return self.memory_module.memory

    @property
    def memory_size(self) -> int:
        return self.memory_module.memory_size

    def inverse_read(
        self,
        input_reprs: torch.Tensor,
        input_mask: torch.Tensor,
        module_name: str,
        detach: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.memory_module.memory

        if detach:
            memory = memory.detach()

        return self.reading_module.inverse_read(
            input_reprs=input_reprs,
            input_mask=input_mask,
            memory=memory,
            module_name=module_name,
        )

    def norm_read(
        self,
        input_reprs: torch.Tensor,
        input_mask: torch.Tensor,
        module_name: str,
        detach: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.memory_module.memory

        if detach:
            memory = memory.detach()

        return self.reading_module.norm_read(
            input_reprs=input_reprs,
            input_mask=input_mask,
            memory=memory,
            module_name=module_name,
        )
