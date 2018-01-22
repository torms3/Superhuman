from __future__ import print_function

import os

import forward
from model import InferenceNet
from options import BaseOptions


def test(opt):
    # Create model.
    net = InferenceNet(opt)
    if opt.chkpt_num > 0:
        net = load_chkpt(net, opt)
    net = net.cuda()
    if not opt.no_eval
        net.eval()

    for dset in params["dsets"]:
        print(dset)
        scanner = make_forward_scanner(dset, **params)
        output = forward.scan(net, scanner)
        save_output(output, dset, **params)


def load_chkpt(model, opt):
    print("LOAD CHECKPOINT: {} iters.".format(opt.chkpt_num))
    fname = os.path.join(opt.model_dir, "model{}.chkpt".format(opt.chkpt_num))
    model.load(fname)
    return model, monitor


if __name__ == "__main__":
    # Options.
    opt = BaseOptions().parse()

    # GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Make directories.
    if not os.path.isdir(opt.exp_dir):
        os.makedirs(opt.exp_dir)
    if not os.path.isdir(opt.fwd_dir):
        os.makedirs(opt.log_dir)

    # Run inference.
    print("Running inference: {}".format(opt.exp_name))
    test(opt)
