from __future__ import print_function

import numpy as np
import time

import torch
from torch.autograd import Variable


class Forward(object):
    """
    Forward scanning.
    """
    def __init__(self, net, opt):
        self.net = net
        self.opt = opt

    def __call__(self, scanner):
        return self._scan(scanner)

    ####################################################################
    ## Non-interface functions.
    ####################################################################

    def _scan(self, scanner):
        elapsed = list()
        start = time.time()
        inputs = scanner.pull()
        while inputs is not None:
            inputs = self._make_variables(inputs)
            # Forward pass.
            outputs = self.net(*inputs)
            scanner.push(self._extract_data(outputs))
            # Elapsed time.
            elapsed.append(time.time() - start)
            print("Elapsed: %.3f s" % elapsed[-1])
            start = time.time()
            # Fetch next inputs.
            inputs = scanner.pull()
        print("Elapsed: %.3f s/patch" % (sum(elapsed)/len(elapsed)))
        print("Throughput: %d voxel/s" % round(scanner.voxels()/sum(elapsed)))
        return scanner.outputs

    def _make_variables(self, sample):
        inputs = list()
        for k in sorted(self.opt.in_spec):
            data = np.expand_dims(sample[k], axis=0)
            tensor = torch.from_numpy(data)
            inputs.append(Variable(tensor, requires_grad=False, volatile=True).cuda())
        return inputs

    def _extract_data(self, outputs):
        outs = dict()
        for i, k in enumerate(sorted(self.opt.out_spec)):
            if k in self.opt.scan_spec:
                scan_channels = self.opt.scan_spec[k][-4]
                narrowed = outputs[i].data.narrow(1, 0, scan_channels)
                outs[k] = np.squeeze(narrowed.cpu().numpy(), axis=(0,))
        return outs
