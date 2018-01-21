from __future__ import print_function

from collections import OrderedDict
import time

import torch
from torch import nn
from torch.autograd import Variable

import loss
from rsunet import RSUNet


class TrainModel(RSUNet):
    """
    RSUNet for training.
    """
    def __init__(self, opt):
        super(TrainModel, self).__init__(opt.in_spec, opt.out_spec, opt.depth)
        self.in_spec = opt.in_spec
        self.out_spec = opt.out_spec
        self.loss_fn = loss.BinaryCrossEntropyWithLogits()

    def forward(self, sample):
        """Runs forward pass and computes loss.

        Args:
            sample (dictionary): Containing inputs, labels, and masks.

        Returns:
            Variable:
                Computed loss.
        """
        # Forward pass.
        inputs = [sample[k] for k in self.in_spec]
        preds = super(TrainModel, self).forward(*inputs)
        # Evaluates loss.
        return self.eval_loss(preds, sample)

    def eval_loss(self, preds, sample):
        self.loss = OrderedDict()
        self.nmsk = OrderedDict()
        for i, k in enumerate(self.out_spec):
            label = sample[k]
            mask  = sample[k + '_mask']
            self.loss[k] = self.loss_fn(preds[i], label, mask)
            self.nmsk[k] = mask.sum()
        return (self.loss.values(), self.nmsk.values())
