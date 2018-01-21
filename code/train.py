from __future__ import print_function

import numpy as np
import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import SNEMI3D_Dataset
import model
from options import BaseOptions
from sampler import get_sampler


def train(opt):
    """
    Training.
    """
    # Create a model.
    net = model.TrainModel(opt)
    if opt.batch_size > 1:
        net = torch.nn.DataParallel(net)
    net = net.cuda()

    # Create a data sampler.
    sampler = get_sampler(opt)
    dataset = SNEMI3D_Dataset(sampler['train'],
                              size=opt.max_iter * opt.batch_size,
                              margin=opt.batch_size * opt.num_workers)
    dataloader = DataLoader(dataset,
                        batch_size=opt.batch_size,
                        shuffle=False,
                        num_workers=opt.num_workers,
                        pin_memory=True)

    # Create an optimizer.
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.base_lr)

    # Profiling.
    fend = list()
    bend = list()

    start = time.time()
    print("======= BEGIN TRAINING LOOP ========")
    for i, sample in enumerate(dataloader):
        sample = make_variables(sample, opt)

        backend = time.time()
        optimizer.zero_grad()
        loss = net(sample).sum()
        loss.backward()
        optimizer.step()
        backend = time.time() - backend

        # Elapsed time.
        elapsed  = time.time() - start
        fend.append(elapsed - backend)
        bend.append(backend)

        # Display.
        print("Iter %6d: loss = %.3f (frontend = %.3f s, backend = %.3f s, elapsed = %.3f s)" % (i+1, loss, fend[i], bend[i], elapsed))
        start = time.time()

    n = opt.max_iter - 10
    print("n = %d, frontend = %.3f s, backend = %.3f s" % (n, sum(fend[-n:])/n, sum(bend[-n:])/n))


def make_variables(sample, opt):
    # Inputs.
    for k in opt.in_spec:
        sample[k] = Variable(sample[k], requires_grad=True).cuda()
    # Labels.
    for k in opt.out_spec:
        sample[k] = Variable(sample[k], requires_grad=False).cuda(async=True)
    # Masks.
    for k in opt.out_spec:
        k +='_mask'
        assert k in sample
        sample[k] = Variable(sample[k], requires_grad=False).cuda(async=True)
    return sample


if __name__ == "__main__":
    # Options.
    opt = BaseOptions().parse()

    # GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Run experiment.
    print("Running experiment: {}".format(opt.exp_name))
    train(opt)
