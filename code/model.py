from __future__ import print_function

from collections import OrderedDict
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import loss
from rsunet import RSUNet


class TrainNet(RSUNet):
    """
    RSUNet for training.
    """
    def __init__(self, opt):
        super(TrainNet, self).__init__(opt.in_spec, opt.out_spec, opt.depth,
                                       opt.momentum)
        self.in_spec = opt.in_spec
        self.out_spec = opt.out_spec
        self.loss_fn = loss.BinaryCrossEntropyWithLogits()

    def forward(self, sample):
        """Runs forward pass and computes loss."""
        # Forward pass.
        inputs = [sample[k] for k in sorted(self.in_spec)]
        preds = super(TrainNet, self).forward(*inputs)
        # Evaluates loss.
        return self.eval_loss(preds, sample)

    def eval_loss(self, preds, sample):
        self.loss = OrderedDict()
        self.nmsk = OrderedDict()
        for i, k in enumerate(self.out_spec):
            label = sample[k]
            mask = sample[k+'_mask'] if k+'_mask' in sample else None
            self.loss[k] = self.loss_fn(preds[i], label, mask)
            self.nmsk[k] = mask.mean()
        return (self.loss.values(), self.nmsk.values())

    def save(self, fpath):
        torch.save(super(TrainNet, self).state_dict(), fpath)

    def load(self, fpath):
        super(TrainNet, self).load_state_dict(torch.load(fpath))


class InferenceNet(RSUNet):
    """
    RSUNet for inference.
    """
    def __init__(self, opt):
        super(InferenceNet, self).__init__(opt.in_spec, opt.out_spec, opt.depth)
        self.in_spec = opt.in_spec
        self.out_spec = opt.out_spec
        self.activation = F.sigmoid

    def forward(self, x):
        preds = super(InferenceNet, self).forward(x)
        return [self.activation(x) for x in preds]

    def load(self, fpath):
        super(InferenceNet, self).load_state_dict(torch.load(fpath))
