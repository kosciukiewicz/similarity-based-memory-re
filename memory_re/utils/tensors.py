from typing import Sequence

import torch
from torch import Tensor, device


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def collate_fn_padding(batch: list[dict[str, torch.Tensor]]):
    padded_batch = {}
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = padded_stack([s[key] for s in batch])

    return padded_batch


def padded_stack(tensors: Sequence[Tensor], padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, tuple(max_shape), fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def extend_tensor(tensor: Tensor, extended_shape: tuple, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[: tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[: tensor_shape[0], : tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[: tensor_shape[0], : tensor_shape[1], : tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[
            : tensor_shape[0], : tensor_shape[1], : tensor_shape[2], : tensor_shape[3]
        ] = tensor

    return extended_tensor


def batch_index(tensor: Tensor, index: Tensor, pad: bool = False) -> Tensor:
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])
