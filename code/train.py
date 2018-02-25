from __future__ import print_function

import numpy as np
import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from dataset import SNEMI3D_Dataset
from model import TrainNet
from monitor import LearningMonitor
from options import BaseOptions
from sampler import get_sampler


def train(opt):
    # cuDNN auto-tuning.
    if opt.autotune:
        torch.backends.cudnn.benchmark = True

    # Create model & learning monitor.
    net = TrainNet(opt)
    monitor = LearningMonitor()
    if opt.chkpt_num > 0:
        net, monitor = load_chkpt(net, monitor, opt)
    model = torch.nn.DataParallel(net) if len(opt.gpu_ids) > 1 else net
    model = model.cuda()
    model.train()

    # Create DataLoaderIters.
    dataiter = prepare_data(opt)

    # Create an optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.base_lr)

    # Create a summary writer.
    writer = SummaryWriter(opt.log_dir)

    # Save model & monitor.
    save_chkpt(net, monitor, opt.chkpt_num, opt)

    start = time.time()
    print("========== BEGIN TRAINING LOOP ==========")
    for i in range(opt.chkpt_num, opt.max_iter):
        sample = make_variables(next(dataiter['train']), opt, phase='train')

        # Optimizer step.
        optimizer.zero_grad()
        losses, nmasks = model(sample)
        loss = sum([l.mean() for l in losses])
        loss.backward()
        optimizer.step()

        # Elapsed time.
        elapsed = time.time() - start

        # Record keeping.
        keys = sorted(opt.out_spec)
        loss = {k: losses[i].mean().data[0] for i, k in enumerate(keys)}
        nmsk = {k: nmasks[i].data[0] for i, k in enumerate(keys)}
        monitoring(monitor, 'train', loss, nmsk, elapsed=elapsed)

        # Averaging & displaying stats.
        if (i+1) % opt.avgs_intv == 0 or i < opt.warm_up:
            average_stats(i+1, monitor, opt, 'train', writer=writer)

        # Validation loop.
        if (i+1) % opt.test_intv == 0:
            validation(i+1, model, dataiter['test'], opt, monitor, writer)

        # Model snapshot.
        if (i+1) % opt.chkpt_intv == 0:
            save_chkpt(net, monitor, i+1, opt)

        # Restart timer.
        start = time.time()

    writer.close()


def validation(iter_num, model, dataiter, opt, monitor, writer):
    # Train -> eval mode.
    if not opt.no_eval:
        model.eval()

    # Runs validation loop.
    print("---------- BEGIN VALIDATION LOOP ----------")
    start = time.time()
    for i in range(opt.test_iter):
        sample = make_variables(next(dataiter), opt, phase='test')
        # Forward pass.
        losses, nmasks = model(sample)
        # Elapsed time.
        elapsed = time.time() - start
        # Monitoring.
        keys = sorted(opt.out_spec)
        loss = {k: losses[i].mean().data[0] for i, k in enumerate(keys)}
        nmsk = {k: nmasks[i].data[0] for i, k in enumerate(keys)}
        monitoring(monitor, 'test', loss, nmsk, elapsed=elapsed)
        # Restart timer.
        start = time.time()

    # Averaging & dispalying.
    average_stats(iter_num, monitor, opt, 'test', writer=writer)
    print("-------------------------------------------")

    # Eval -> train mode.
    model.train()


def monitoring(monitor, phase, losses, nmasks, **kwargs):
    assert phase in ['train','test']
    monitor.add_to_num(losses, phase)
    monitor.add_to_denom(nmasks, phase)
    for k, v in kwargs.items():
        monitor.add_to_num({k: v}, phase)
        monitor.add_to_denom({k: 1}, phase)


def logging(writer, phase, iter_num, **kwargs):
    assert phase in ['train','test']
    for k, v in kwargs.items():
        writer.add_scalar('{}/{}'.format(phase, k), v, iter_num)


def average_stats(iter_num, monitor, opt, phase, writer=None):
    assert phase in ['train','test']

    # Averaging stats.
    monitor.compute_avgs(iter_num, phase)
    loss = dict()
    for k in sorted(opt.out_spec):
        loss[k] = monitor.get_last_value(k, phase)
    elapsed = monitor.get_last_value('elapsed', phase)

    # Logging to the event logs (optional).
    if writer is not None:
        logging(writer, phase, iter_num, elapsed=elapsed, **loss)

    # Dispaly to console.
    disp = "[%s] Iter: %8d, " % (phase, iter_num)
    for k, v in loss.items():
        disp += "%s = %.3f, " % (k, v)
    disp += "lr = %.6f, " % opt.base_lr
    disp += "(elapsed = %.3f). " % elapsed
    print(disp)


def make_variables(sample, opt, phase):
    assert phase in ['train','test']
    requires_grad = (phase == 'train')
    volatile = (phase == 'test')
    # Inputs.
    for k in opt.in_spec:
        sample[k] = Variable(sample[k], requires_grad=requires_grad, volatile=volatile).cuda()
    # Labels.
    for k in opt.out_spec:
        sample[k] = Variable(sample[k], requires_grad=False).cuda(async=True)
    # Masks.
    for k in opt.out_spec:
        k +='_mask'
        assert k in sample
        sample[k] = Variable(sample[k], requires_grad=False).cuda(async=True)
    return sample


def prepare_data(opt):
    sampler = get_sampler(opt)
    # Dataset.
    dataset_size = (opt.max_iter - opt.chkpt_num) * opt.batch_size
    dataset = dict()
    dataset['train'] = SNEMI3D_Dataset(sampler['train'],
                            size=dataset_size,
                            margin=opt.batch_size * opt.num_workers)
    dataset['test'] = SNEMI3D_Dataset(sampler['val'],
                          size=dataset_size,
                          margin=opt.batch_size * opt.num_workers)
    # DataLoader.
    dataloader = dict()
    dataloader['train'] = DataLoader(dataset['train'],
                              batch_size=opt.batch_size,
                              shuffle=False,
                              num_workers=opt.num_workers,
                              pin_memory=True)
    dataloader['test'] = DataLoader(dataset['test'],
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True)
    # DataLoader iterator.
    dataiter = dict()
    dataiter['train'] = iter(dataloader['train'])
    dataiter['test'] = iter(dataloader['test'])
    return dataiter


def load_chkpt(model, monitor, opt):
    print("LOAD CHECKPOINT: {} iters.".format(opt.chkpt_num))

    # Load model.
    fname = os.path.join(opt.model_dir, "model{}.chkpt".format(opt.chkpt_num))
    model.load(fname)

    # Load learning monitor.
    fname = os.path.join(opt.log_dir, "stats{}.h5".format(opt.chkpt_num))
    monitor.load(fname)

    return model, monitor


def save_chkpt(model, monitor, chkpt_num, opt):
    print("SAVE CHECKPOINT: {} iters.".format(chkpt_num))

    # Save model.
    fname = os.path.join(opt.model_dir, "model{}.chkpt".format(chkpt_num))
    model.save(fname)

    # Save learning monitor.
    fname = os.path.join(opt.log_dir, "stats{}.h5".format(chkpt_num))
    monitor.save(fname, chkpt_num)


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
