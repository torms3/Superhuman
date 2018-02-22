from __future__ import print_function

import argparse
import os


class BaseOptions(object):
    """
    Base options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_dir', required=True)
        self.parser.add_argument('--exp_name', required=True)

        # Training.
        self.parser.add_argument('--base_lr', type=float, default=0.01)
        self.parser.add_argument('--max_iter', type=int, default=1000000)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--num_workers', type=int, default=1)
        self.parser.add_argument('--gpu_ids', type=str, default=['0'], nargs='+')
        self.parser.add_argument('--test_intv', type=int, default=1000)
        self.parser.add_argument('--test_iter', type=int, default=100)
        self.parser.add_argument('--avgs_intv', type=int, default=50)
        self.parser.add_argument('--warm_up', type=int, default=50)
        self.parser.add_argument('--chkpt_intv', type=int, default=10000)
        self.parser.add_argument('--chkpt_num', type=int, default=0)
        self.parser.add_argument('--no_eval', action='store_true')
        self.parser.add_argument('--size_average', action='store_true')

        # Model spec.
        self.parser.add_argument('--fov', type=int, default=[32,160,160], nargs='+')
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--momentum', type=float, default=0.1)

        # Data augmentation & transform.
        self.parser.add_argument('--long_range', action='store_true')
        self.parser.add_argument('--misalign', action='store_true')
        self.parser.add_argument('--missing', action='store_true')
        self.parser.add_argument('--blur', action='store_true')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Model spec.
        opt.fov = tuple(opt.fov)
        opt.in_spec = dict(input=(1,) + opt.fov)
        if opt.long_range:
            opt.out_spec = dict(affinity=(12,) + opt.fov)
        else:
            opt.out_spec = dict(affinity=(3,) + opt.fov)

        # Data augmentation.
        opt.aug = [opt.misalign, opt.missing, opt.blur]

        # Directories.
        opt.exp_dir = 'experiments/{}'.format(opt.exp_name)
        opt.log_dir = os.path.join(opt.exp_dir, 'logs')
        opt.model_dir = os.path.join(opt.exp_dir, 'models')

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt


class TestOptions(object):
    """
    Test options.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--exp_name', required=True)
        self.parser.add_argument('--data_dir', default="")
        self.parser.add_argument('--data_names', nargs='+')
        self.parser.add_argument('--data_prefix', default="SNEMI3D")
        self.parser.add_argument('--data_tag', default="")
        self.parser.add_argument('--gpu_ids', type=str, default=['0'], nargs='+')
        self.parser.add_argument('--chkpt_num', type=int, default=0)
        self.parser.add_argument('--no_eval', action='store_true')

        # Model spec.
        self.parser.add_argument('--fov', type=int, default=[32,160,160], nargs='+')
        self.parser.add_argument('--depth', type=int, default=4)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('--scan_channels', type=int, default=3)
        self.parser.add_argument('--no_BN', action='store_true')

        # For benchmark.
        self.parser.add_argument('--dummy', action='store_true')
        self.parser.add_argument('--input_size', type=int, default=[128,1024,1024], nargs='+')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        # Model spec.
        opt.fov = tuple(opt.fov)
        opt.in_spec = dict(input=(1,) + opt.fov)
        opt.out_spec = dict(affinity=(opt.out_channels,) + opt.fov)

        # Scan spec.
        opt.scan_spec = dict(affinity=(opt.scan_channels,) + opt.fov)
        opt.scan_params = dict(stride=(0.5,0.5,0.5), blend='bump')

        # Directories.
        opt.exp_dir = 'experiments/{}'.format(opt.exp_name)
        opt.model_dir = os.path.join(opt.exp_dir, 'models')
        opt.fwd_dir = os.path.join(opt.exp_dir, 'forward')

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt
