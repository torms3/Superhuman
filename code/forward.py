from __future__ import print_function

import numpy as np
import time

import torch


def scan(net, scanner):
    with torch.no_grad():
        start = time.time()
        inputs = scanner.pull()
        while inputs is not None:
            inputs = make_variables(inputs)
            # Forward pass.
            outputs = net(*inputs)
            push_outputs(scanner, outputs, opt)
            # Elapsed time.
            elapsed = time.time() - start
            print("Elapsed: %.3f" % elapsed)
            start = time.time()
            # Fetch next inputs.
            inputs = scanner.pull()
    return scanner


def make_variables(sample, opt):
    inputs = list()
    for k in sorted(opt.in_spec):
        inputs.append(Variable(sample[k], requires_grad=False).cuda())
    return inputs


def push_outputs(scanner, outputs, opt):
    outs = dict()
    for i, k in enumerate(sorted(opt.out_spec)):
        if k in opt.scan_spec:
            outs[k] = extract_data(outputs[i])
    scanner.push(outs)


def extract_data(expanded_variable):
    return np.squeeze(expanded_variable.data.cpu().numpy(), axis=(0,))
