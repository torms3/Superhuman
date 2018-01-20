from __future__ import print_function

import time

import torch
from torch import nn
from torch.autograd import Variable

import loss
from rsunet import RSUNet


class Model(object):

    def __init__(self, opt):
        # Options.
        self.opt = opt

        # In/out spec.
        self.in_spec = opt.in_spec
        self.out_spec = opt.out_spec

        # Create a net.
        net = RSUNet(opt.in_spec, opt.out_spec, opt.depth)
        if opt.batch_size > 1:
            net = torch.nn.DataParallel(net)
        self.net = net.cuda()

        # Create an optimizer and a loss function.
        self.optimizer = torch.optim.Adam(net.parameters(), lr=opt.base_lr)
        self.loss_fn = loss.BinaryCrossEntropyWithLogits()

    def step(self, i, sample):
        t0 = time.time()
        sample = self.make_variables(sample)
        t1 = time.time()
        preds = self.net(*[sample[k] for k in self.in_spec])
        losses, nmsks = self.eval_loss(preds, sample)
        self.update_model(losses)
        t2 = time.time()
        # self.log_errors(losses, nmsks)
        # avg_loss = dict([(k, v.data[0]/nmsks[k]) for k,v in losses.items()])
        avg_loss = sum(losses.values())/sum(nmsks.values())
        disp  = "Iter %6d: " % (i+1)
        disp += "loss = %.3f " % avg_loss
        disp += "(frontend = %.3f s, backend = %.3f s, elapsed = %.3f s)" % (t1-t0, t2-t1, t2-t0)
        print(disp)

    def make_variables(self, sample):
        # Inputs.
        for k in self.in_spec:
            sample[k] = Variable(sample[k], requires_grad=True).cuda()
        # Labels.
        for k in self.out_spec:
            sample[k] = Variable(sample[k], requires_grad=False).cuda(async=True)
        # Masks.
        for k in self.out_spec:
            k +='_mask'
            assert k in sample
            sample[k] = Variable(sample[k], requires_grad=False).cuda(async=True)
        return sample

    def eval_loss(self, preds, sample):
        losses, nmasks = dict(), dict()
        for k in self.out_spec.keys():
            assert k+'_mask' in sample
            mask = sample[k+'_mask']
            losses[k] = self.loss_fn(preds[k], sample[k], mask=mask)
            nmasks[k] = mask.sum()
        return losses, nmasks

    def update_model(self, losses):
        self.optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()

    # def log_errors(self, losses, nmasks):
