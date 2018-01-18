from __future__ import print_function

import numpy as np
import os

import dataprovider as dp
import dataprovider.emio as emio


def get_sampler(opt):
    data_dir = os.path.expanduser(opt.data_dir)
    # Dataset.
    data = dict()
    # Image.
    fname = 'train_img.h5'
    data['img'] = emio.imread(os.path.join(data_dir, fname)).astype('float32')
    data['img'] /= 255.0
    # Segmentation.
    fname = 'train_seg.h5'
    data['seg'] = emio.imread(os.path.join(data_dir, fname))
    # Mask for training.
    fname = 'train_msk.h5'
    data['msk_train'] = emio.imread(os.path.join(data_dir, fname))
    # Mask for validation.
    fname = 'val_msk.h5'
    data['msk_val'] = emio.imread(os.path.join(data_dir, fname))
    # Samplers.
    sampler = dict()
    kwargs = {'fov':opt.fov, 'long_range':opt.long_range, 'aug':opt.aug}
    sampler['train'] = Sampler(data, 'train', **kwargs)
    sampler['val'] = Sampler(data, 'val', **kwargs)
    return sampler


class Sampler(object):
    """
    Data sampler.
    """
    def __init__(self, data, mode, fov=(32,160,160), long_range=True, aug=[0,0,0]):
        assert mode in ['train','val']
        # Data dictionary.
        self.data = data
        # Sample spec.
        fov = tuple([x+1 for x in fov])  # Will be cropped later.
        self.spec = dict(input=fov, affinity=fov, affinity_mask=fov)
    	# Augmentation.
    	self.aug = aug
        # Long-range affinity.
    	self.long_range = long_range
    	# Mode.
    	self.mode = mode
        # Build data providers.
        self.dp = self.build_data_provider()

    def __call__(self, **kwargs):
        return self.dp('random', **kwargs)

    ####################################################################
    ## Builders.
    ####################################################################

    def build_data_provider(self):
        vdp = dp.VolumeDataProvider()
        vd = self.build_dataset()
        vdp.add_dataset(vd)
        vdp.set_sampling_weights()
        vdp.set_augmentor(self._aug())
        vdp.set_postprocessor(self._post())
        return vdp

    def build_dataset(self):
        # Image.
        img = self.data['img']
        # Segmentation.
        seg = self.data['seg']
        # Mask.
        msk = self.data['msk_' + self.mode]
        # Build dataset.
        vd = dp.VolumeDataset()
        vd.add_raw_data(key='input',         data=img)
        vd.add_raw_data(key='affinity',      data=seg)
        vd.add_raw_mask(key='affinity_mask', data=msk, loc=True)
        vd.set_spec(self.spec)
        return vd

    def _aug(self):
        aug = dp.Augmentor()
        if self.aug[0]:
            print("[misalign]")
            aug.append('misalign', max_trans=17.0)
        if self.aug[1]:
            print("[missing] max_sec = {}".format(10))
            aug.append('missing', max_sec=10, mode='mix', random_color=True)
        if self.aug[2]:
            print("[blur] max_sec = {}".format(10))
            aug.append('blur', max_sec=10, mode='mix')
        if self.mode == 'train':
            aug.append('warp')
            aug.append('greyscale', mode='mix')
            aug.append('flip')
        return aug

    def _post(self):
        post = dp.Transformer()
        dst = list()
        dst.append((0,0,1))
        dst.append((0,1,0))
        dst.append((1,0,0))
        if self.long_range:
            dst.append((0,0,3))
            dst.append((0,3,0))
            dst.append((2,0,0))
            dst.append((0,0,9))
            dst.append((0,9,0))
            dst.append((3,0,0))
            dst.append((0,0,27))
            dst.append((0,27,0))
            dst.append((4,0,0))
        aff = dp.Affinity(dst, 'affinity', 'affinity', crop=(1,1,1), base_w=0.5)
        post.append(aff)
        return post


if __name__ == "__main__":

    import h5py
    import time

    from options import BaseOptions

    # Options.
    opt = BaseOptions().parse()

    print("Creating data samplers...")
    sampler = get_sampler(opt)

    # 10 random samples.
    print("Sampling...")
    for i in range(opt.max_iter):
        start = time.time()
        sample = sampler['train'](imgs=['input'])
        print("Elapsed: %.3f" % (time.time() - start))
        # fname = 'sample%.2d.h5' % (i+1)
        # print("Save as {}...".format(fname))
        # if os.path.exists(fname):
        #     os.remove(fname)
        # f = h5py.File(fname)
        # for name, data in sample.items():
        #     f.create_dataset('/' + name, data=data)
        # f.close()
