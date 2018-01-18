#!/usr/bin/env python
__doc__ = """

Training.

Kisuk Lee <kisuklee@mit.edu>, 2017
"""

import numpy as np
import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import SNEMI3D_Dataset
import loss
import model
from options import BaseOptions
from sampler import get_sampler


def train(opt):
    """
    Training.
    """
    # Create a net.
    net = model.Model(opt.in_spec, opt.out_spec, opt.depth).cuda()

    # Create a data sampler.
    sampler = get_sampler(opt)
    dataset = SNEMI3D_Dataset(sampler['train'], size=opt.max_iter)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, pin_memory=True)

    # Create an optimizer and a loss function.
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.base_lr)
    loss_fn = loss.BinaryCrossEntropyWithLogits()

    start = time.time()
    print("======= BEGIN TRAINING LOOP ========")
    for i, sample in enumerate(dataloader):
        t0 = time.time()
        inputs, labels, masks = make_variables(sample, opt)
        t1 = time.time()

        # Running forward pass.
        backend = time.time()
        preds = net(*inputs)
        losses, nmsks = eval_loss(opt.out_spec, preds, labels, masks, loss_fn)
        update_model(optimizer, losses)
        backend = time.time() - backend

        # Elapsed time.
        elapsed  = time.time() - start
        avg_loss = sum(losses.values())/sum(nmsks.values())
        print("Iter %6d: loss = %.3f (variable = %.3f s, backend = %.3f s, elapsed = %.3f s)" % (i+1, avg_loss, t1-t0, backend, elapsed))
        start = time.time()


def update_model(optimizer, losses):
    """Runs the backward pass and updates model parameters."""
    optimizer.zero_grad()
    total_loss = sum(losses.values())
    total_loss.backward()
    optimizer.step()


def eval_loss(out_spec, preds, labels, masks, loss_fn):
    """
    Evaluates the error of the predictions according to the available
    labels and masks.

    Assumes labels are ordered according to the sample_spec.
    """
    assert len(masks) == len(labels), "Mismatched masks and labels"
    assert len(preds) == len(labels), "Mismatched preds and labels"

    losses = dict()
    nmasks = dict()

    for i, k in enumerate(out_spec.keys()):
        losses[k] = loss_fn(preds[i], labels[i], masks[i])
        nmasks[k] = masks[i].sum()

    return losses, nmasks


def make_variable(np_arr, requires_grad=True, volatile=False):
    """Creates a torch.autograd.Variable from a np array."""
    if not volatile:
        return Variable(torch.from_numpy(np_arr.copy()), requires_grad=requires_grad).cuda()
    else:
        return Variable(torch.from_numpy(np_arr.copy()), volatile=True).cuda()

def make_variables(sample, opt):
    """Creates the Torch variables for a sample."""
    inputs = opt.in_spec.keys()
    labels = opt.out_spec.keys()
    masks  = [l + '_mask' for l in labels]

    input_vars = [Variable(sample[k], requires_grad=True).cuda() for k in inputs]
    label_vars = [Variable(sample[k], requires_grad=False).cuda(async=True) for k in labels]
    mask_vars  = [Variable(sample[k], requires_grad=False).cuda(async=True) for k in masks]

    return input_vars, label_vars, mask_vars


if __name__ == "__main__":
    # Options.
    opt = BaseOptions().parse()

    # GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Run experiment.
    print("Running experiment: {}".format(opt.exp_name))
    train(opt)
