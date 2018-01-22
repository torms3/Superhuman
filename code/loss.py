import torch
from torch import nn
from torch.nn import functional as F


class BinaryCrossEntropyWithLogits(nn.Module):
    """
    A version of BCE w/ logits with the ability to mask out regions of output.
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input, target, mask=None):
        # By setting size_average=False,
        # we compute loss.sum() rather than loss.mean().
        return F.binary_cross_entropy_with_logits(input, target, weight=mask)
