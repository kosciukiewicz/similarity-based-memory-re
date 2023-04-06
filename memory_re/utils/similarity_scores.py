import torch
from torch import Tensor, nn


def get_cos_sim(a: Tensor, b: Tensor) -> Tensor:
    a_norm = nn.functional.normalize(a, p=2, dim=1)
    b_norm = nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
