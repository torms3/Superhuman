import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class BCELoss(nn.Module):
    """
    A version of BCE w/ logits with the ability to mask out regions of output.
    """
    def __init__(self, size_average=False):
        super(BCELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, mask=None):
        loss = F.binary_cross_entropy_with_logits(input, target, weight=mask,
                    size_average=False)

        if mask is not None:
            nmsk = torch.nonzero(mask.data).size(0)
        else:
            nmsk = torch.numel(mask.data)

        # Pixelwise averge or sum.
        if self.size_average:
            return loss/nmsk, Variable(torch.Tensor([1])).cuda()
        else:
            return loss, Variable(torch.Tensor([nmsk])).cuda()
