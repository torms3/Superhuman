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

        self.parser.add_argument('--base_lr', type=float, default=0.01)
        self.parser.add_argument('--max_iter', type=int, default=1000000)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--gpu_ids', type=str, default=['0'], nargs='+')

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
        opt.fov = (32,160,160)
        opt.depth = 4
        opt.in_spec = dict(input=(1,) + opt.fov)
        if opt.long_range:
            opt.out_spec = dict(affinity=(12,) + opt.fov)
        else:
            opt.out_spec = dict(affinity=(3,) + opt.fov)

        # Data augmentation.
        opt.aug = [opt.misalign, opt.missing, opt.blur]

        args = vars(opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        self.opt = opt
        return self.opt
