from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_noam_warmup_scheduler(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    def lr_lambda(current_step: int):
        current_step += 1
        return min(current_step ** (-0.5), current_step * num_warmup_steps ** (-1.5))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
