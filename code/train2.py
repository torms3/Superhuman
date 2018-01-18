from __future__ import print_function

import numpy as np
import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from async import AsyncSampler
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
    train_sampler = AsyncSampler(sampler['train'])
    val_sampler = AsyncSampler(sampler['val'])

    # Create an optimizer and a loss function.
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.base_lr)
    loss_fn = loss.BinaryCrossEntropyWithLogits()

    start = time.time()
    print("======= BEGIN TRAINING LOOP ========")
    for i in range(opt.max_iter):
        t0 = time.time()
        sample = train_sampler()
        t1 = time.time()
        inputs, labels, masks = make_variables(sample, opt)
        t2 = time.time()

        # Running forward pass.
        backend = time.time()
        preds = net(*inputs)
        losses, nmsks = eval_loss(opt.out_spec, preds, labels, masks, loss_fn)
        update_model(optimizer, losses)
        backend = time.time() - backend

        # Elapsed time.
        elapsed  = time.time() - start
        avg_loss = sum(losses.values())/sum(nmsks.values())
        print("Iter %6d: loss = %.3f (fetch: %.3fs, variable: %.3fs, backend: %.3fs, elapsed: %.3fs)" % (i+1, avg_loss, t1-t0, t2-t1, backend, elapsed))
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
        return Variable(torch.from_numpy(np_arr), requires_grad=requires_grad).cuda()
    else:
        return Variable(torch.from_numpy(np_arr), volatile=True).cuda()


def make_variables(sample, opt):
    """Creates the Torch variables for a sample."""
    inputs = opt.in_spec.keys()
    labels = opt.out_spec.keys()
    masks  = [l + '_mask' for l in labels]

    input_vars = [make_variable(sample[k], requires_grad=True)  for k in inputs]
    label_vars = [make_variable(sample[k], requires_grad=False) for k in labels]
    mask_vars  = [make_variable(sample[k], requires_grad=False) for k in masks]

    return input_vars, label_vars, mask_vars


if __name__ == "__main__":
    # Options.
    opt = BaseOptions().parse()

    # GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Run experiment.
    print("Running experiment: {}".format(opt.exp_name))
    train(opt)
