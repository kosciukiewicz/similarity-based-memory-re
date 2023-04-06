import torch

from memory_re.models.memory_re.memory_modules.base import BaseMemoryReadingModule


class CosineMemoryReadingModule(BaseMemoryReadingModule):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self._cos = torch.nn.CosineSimilarity(dim=1, eps=eps)

    def inverse_read(
        self,
        input_reprs: torch.Tensor,
        input_mask: torch.Tensor,
        memory: torch.Tensor,
        module_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_reprs.shape[0]
        attn = torch.zeros(
            input_reprs.shape[0],
            input_reprs.shape[1],
            memory.shape[0],
            dtype=torch.float,
            device=input_reprs.device,
        )
        for b in range(batch_size):
            batch_input_reprs = input_reprs[b]
            a_norm = torch.nn.functional.normalize(batch_input_reprs, p=2, dim=1)
            b_norm = torch.nn.functional.normalize(memory, p=2, dim=1)

            attn[b] = torch.mm(a_norm, b_norm.transpose(0, 1))
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
        batch_size = input_reprs.shape[0]
        attn = torch.zeros(
            input_reprs.shape[0],
            input_reprs.shape[1],
            memory.shape[0],
            dtype=torch.float,
            device=input_reprs.device,
        )

        for b in range(batch_size):
            batch_input_reprs = input_reprs[b].squeeze()
            a_n, b_n = batch_input_reprs.norm(dim=1)[:, None], memory.norm(dim=1)[:, None]
            a_norm = batch_input_reprs / torch.max(a_n, self._eps * torch.ones_like(a_n))
            b_norm = memory / torch.max(b_n, self._eps * torch.ones_like(b_n))

            attn[b] = torch.mm(a_norm, b_norm.transpose(0, 1))
        attn[~input_mask.bool()] = -1e25

        return torch.matmul(attn, memory), attn
