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
        with torch.no_grad():
            start = time.time()
            inputs = scanner.pull()
            while inputs is not None:
                inputs = self._make_variables(inputs)
                # Forward pass.
                outputs = self.net(*inputs)
                self._push_outputs(scanner, outputs)
                # Elapsed time.
                elapsed = time.time() - start
                print("Elapsed: %.3f" % elapsed)
                start = time.time()
                # Fetch next inputs.
                inputs = scanner.pull()
        return scanner.outputs

    def _make_variables(self, sample):
        inputs = list()
        for k in sorted(self.opt.in_spec):
            data = np.expand_dims(sample[k], axis=0)
            tensor = torch.from_numpy(data)
            inputs.append(Variable(tensor, requires_grad=False).cuda())
        return inputs

    def _push_outputs(self, scanner, outputs):
        outs = dict()
        for i, k in enumerate(sorted(self.opt.out_spec)):
            if k in self.opt.scan_spec:
                outs[k] = self._extract_data(outputs[i])
        scanner.push(outs)

    def _extract_data(self, expanded_variable):
        return np.squeeze(expanded_variable.data.cpu().numpy(), axis=(0,))
