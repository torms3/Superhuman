from __future__ import print_function

import numpy as np
import os

import dataprovider as dp
import dataprovider.emio as emio

from forward import Forward
from model import InferenceNet
from options import TestOptions


def test(opt):
    # Create model.
    net = InferenceNet(opt)
    if opt.chkpt_num > 0:
        net = load_chkpt(net, opt)
    net = net.cuda()
    if not opt.no_eval:
        net.eval()

    # Forward scan.
    fwd = Forward(net, opt)
    for data_name in opt.data_names:
        print(data_name)
        scanner = make_forward_scanner(data_name, opt)
        output = fwd(scanner)
        save_output(data_name, output, opt)


def load_chkpt(model, opt):
    print("LOAD CHECKPOINT: {} iters.".format(opt.chkpt_num))
    fname = os.path.join(opt.model_dir, "model{}.chkpt".format(opt.chkpt_num))
    model.load(fname)
    return model


def make_forward_scanner(data_name, opt):
    # Read an EM image.
    img = emio.imread(os.path.join(opt.data_dir, data_name + '_img.h5'))
    img = (img/255.).astype('float32')
    # Build Dataset.
    vd = dp.VolumeDataset()
    vd.add_raw_data(key='input', data=img)
    vd.set_spec(opt.in_spec)
    return dp.ForwardScanner(vd, opt.scan_spec, params=opt.scan_params)


def save_output(data_name, output, opt):
    for k in output.data:
        data = output.get_data(k)
        prefix = "" if not opt.data_prefix else opt.data_prefix + '_'
        tag = "" if not opt.data_tag else '_' + opt.data_tag
        base_name = "{}{}_{}_{}{}.h5".format(prefix, data_name, k, opt.chkpt_num, tag)
        full_name = os.path.join(opt.fwd_dir, base_name)
        emio.imsave(data, full_name)


if __name__ == "__main__":
    # Options.
    opt = TestOptions().parse()

    # GPUs.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt.gpu_ids)

    # Make directories.
    if not os.path.isdir(opt.exp_dir):
        os.makedirs(opt.exp_dir)
    if not os.path.isdir(opt.model_dir):
        os.makedirs(opt.model_dir)
    if not os.path.isdir(opt.fwd_dir):
        os.makedirs(opt.fwd_dir)

    # Run inference.
    print("Running inference: {}".format(opt.exp_name))
    test(opt)
