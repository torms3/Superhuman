from __future__ import print_function

import numpy as np
import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from dataset import SNEMI3D_Dataset
from model import TrainModel
from options import BaseOptions
from sampler import get_sampler


def train(opt):
    # Create a model.
    net = TrainModel(opt)
    if opt.batch_size > 1:
        net = torch.nn.DataParallel(net)
    net = net.cuda()

    # Create DataLoaders.
    sampler = get_sampler(opt)
    dataset = dict()
    dataloader = dict()
    dataset['train'] = SNEMI3D_Dataset(sampler['train'],
                            size=opt.max_iter * opt.batch_size,
                            margin=opt.batch_size * opt.num_workers)
    dataloader['train'] = DataLoader(dataset['train'],
                              batch_size=opt.batch_size,
                              shuffle=False,
                              num_workers=opt.num_workers,
                              pin_memory=True)
    dataset['val'] = SNEMI3D_Dataset(sampler['val'],
                          size=opt.max_iter * opt.batch_size,
                          margin=opt.batch_size * opt.num_workers)
    dataloader['val'] = DataLoader(dataset['val'],
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True)

    # Create an optimizer.
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.base_lr)

    # Create a summary writer.
    writer = SummaryWriter(opt.log_dir)

    # Save model.
    print("SAVE CHECKPOINT: {} iters.".format(0))
    fname = os.path.join(opt.model_dir, 'model{}.chkpt'.format(0))
    torch.save(net.state_dict(), fname)

    start = time.time()
    print("======= BEGIN TRAINING LOOP ========")
    for i, sample in enumerate(dataloader['train']):
        sample = make_variables(sample, opt, phase='train')

        # Optimizer step.
        backend = time.time()
        optimizer.zero_grad()
        losses, nmasks = net(sample)
        loss = sum([l.sum() for l in losses])
        nmsk = sum([n.sum() for n in nmasks])
        loss.backward()
        optimizer.step()
        backend = time.time() - backend

        # Elapsed time.
        elapsed = time.time() - start

        # Logging.
        logging(i+1, losses, nmasks, elapsed, opt, writer, 'train')

        # Validation loop.
        if (i+1) % opt.test_intv == 0:
            evaluation(i+1, net, dataloader['val'], opt, writer)

        # Model snapshot.
        if (i + 1) % opt.checkpoint == 0:
            print("SAVE CHECKPOINT: {} iters.".format(i+1))
            fname = os.path.join(opt.model_dir, 'model{}.chkpt'.format(i+1))
            torch.save(net.state_dict(), fname)

        start = time.time()

    writer.close()


def evaluation(iter_num, model, dataloader, opt, writer):
    model.eval()
    start = time.time()
    dataloader = iter(dataloader)
    for i in range(opt.test_iter):
        sample = make_variables(next(dataloader), opt, phase='eval')

        # Optimizer step.
        optimizer.zero_grad()
        losses, nmasks = net(sample)
        loss = sum([l.sum() for l in losses])
        nmsk = sum([n.sum() for n in nmasks])
        loss = loss.sum()
        loss.backward()
        optimizer.step()

        # Elapsed time.
        elapsed = time.time() - start

        # Logging.
        writer.add_scalar('val/loss', loss/nmsk, i)
        writer.add_scalar('val/elapsed', elapsed, i)
    model.train()


def logging(iter_num, losses, nmasks, elapsed, opt, writer, phase):
    disp = "Iter %8d: " % iter_num
    for i, k in enumerate(sorted(opt.out_spec)):
        loss = (losses[i].sum()/nmasks[i].sum()).data[0]
        disp += "%s = %.3f " % (k, loss)
        writer.add_scalar('{}/{}'.format(phase, k), loss, iter_num)
    disp += "(elapsed = %.3f s)" % elapsed
    writer.add_scalar('{}/elapsed'.format(phase, k), elapsed, iter_num)
    print(disp)


def make_variables(sample, opt, phase):
    assert phase in ['train','eval']
    requires_grad = (phase == 'train')
    # Inputs.
    for k in opt.in_spec:
        sample[k] = Variable(sample[k], requires_grad=requires_grad).cuda()
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

    # Make directories.
    if not os.path.isdir(opt.exp_dir):
        os.makedirs(opt.exp_dir)
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.isdir(opt.model_dir):
        os.makedirs(opt.model_dir)

    # Run experiment.
    print("Running experiment: {}".format(opt.exp_name))
    train(opt)
