#!/usr/bin/env python
__doc__ = """

Symmetric 3D U-Net.

(Optional)
Residual skip connections.

Kisuk Lee <kisuklee@mit.edu>, 2017
Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""

import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


# Global switches.
residual = True
upsample = 'bilinear'
use_bn   = True

# Number of feature maps.
nfeatures = [24,32,48,72,104,144]

# Filter size.
sizes = [(3,3,3)] * len(nfeatures)

# In/out embedding.
embed_ks = (1,5,5)
embed_nin = nfeatures[0]
embed_nout = embed_nin


def pad_size(ks, mode):
    assert mode in ['valid', 'same', 'full']
    if mode == 'valid':
        pad = (0,0,0)
    elif mode == 'same':
        assert all([x %  2 for x in ks])
        pad = tuple(x // 2 for x in ks)
    elif mode == 'full':
        pad = tuple(x - 1 for x in ks)
    return pad


def batchnorm(D_out):
    i = lambda x: x
    return nn.BatchNorm3d(D_out, eps=1e-05, momentum=0.001) if use_bn else i


def residual_sum(x, skip):
    i = lambda x: x
    return x + skip if residual else i


class Conv(nn.Module):
    """
    3D convolution w/ MSRA init.
    """
    def __init__(self, D_in, D_out, ks, st, pd, bias=True):
        nn.Module.__init__(self)
        self.conv = nn.Conv3d(D_in, D_out, ks, st, pd, bias=bias)
        init.kaiming_normal(self.conv.weight)
        if bias:
            init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ConvT(nn.Module):
    """
    3D convolution transpose w/ MSRA init.
    """
    def __init__(self, D_in, D_out, ks, st, pd=(0,0,0), bias=True):
        nn.Module.__init__(self)
        self.conv = nn.ConvTranspose3d(D_in, D_out, ks, st, pd, bias=bias)
        init.kaiming_normal(self.conv.weight)
        if bias:
            init.constant(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ConvMod(nn.Module):
    """
    Convolution module.
    """
    def __init__(self, D_in, D_out, ks, activation=F.elu):
        nn.Module.__init__(self)
        # Convolution params.
        st = (1,1,1)
        pd = pad_size(ks, "same")
        bias = not use_bn
        # Convolution.
        self.conv1 = Conv(D_in,  D_out, ks, st, pd, bias)
        self.conv2 = Conv(D_out, D_out, ks, st, pd, bias)
        self.conv3 = Conv(D_out, D_out, ks, st, pd, bias)
        # Batch normalization.
        self.bn1 = batchnorm(D_out)
        self.bn2 = batchnorm(D_out)
        self.bn3 = batchnorm(D_out)
        # Activation function.
        self.activation = activation

    def forward(self, x):
        # Conv 1.
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        skip = out
        # Conv 2.
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        # Conv 3.
        out = self.conv3(out)
        out = residual_sum(out,skip)
        out = self.bn3(out)
        out = self.activation(out)
        return out


class UpsampleMod(nn.Module):
    """
    Transposed Convolution module.
    """
    def __init__(self, D_in, D_out, up=(1,2,2), mode="bilinear", activation=F.elu):
        nn.Module.__init__(self)

        if mode == "bilinear":
            self.up = nn.Upsample(scale_factor=up, mode="trilinear")
            self.conv = Conv(D_in, D_out, (1,1,1), (1,1,1), (0,0,0))
        elif mode == "nearest":
            self.up = nn.Upsample(scale_factor=up, mode="nearest")
            self.conv = Conv(D_in, D_out, (1,1,1), (1,1,1), (0,0,0))
        elif mode == "transpose":
            self.up = ConvT(D_in, D_out, ks=up, st=up, bias=True)
            self.conv = lambda x: x
        else:
            assert False

        self.bn = batchnorm(D_out)
        self.activation = activation

    def forward(self, x, skip):
        return self.activation(self.bn(self.conv(self.up(x)) + skip))


class Conv1(nn.Module):
    """
    Single convolution module.
    """
    def __init__(self, D_in, D_out, ks, st=(1,1,1), activation=F.elu):
        nn.Module.__init__(self)
        pd = pad_size(ks, "same")
        self.conv = Conv(D_in, D_out, ks, st, pd, bias=True)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))


class OutputMod(nn.Module):
    """
    Embedding -> Output module.
    """
    def __init__(self, D_in, out_spec, ks=(1,1,1), st=(1,1,1)):
        """out_spec should be an Ordered Dict."""
        nn.Module.__init__(self)
        pd = pad_size(ks, "same")
        self.output_layers = []
        for k, v in out_spec.items():
            D_out = v[0]
            setattr(self, k, Conv(D_in, D_out, ks, st, pd, bias=True))
            self.output_layers.append(k)

    def forward(self, x):
        return [getattr(self,layer)(x) for layer in self.output_layers]


class Model(nn.Module):
    """
    Full model.
    """
    def __init__(self, in_spec, out_spec, depth, **kwargs):
        nn.Module.__init__(self)

        # Model assumes a single input.
        assert len(in_spec)==1, "model takes a single input"
        D_in = list(in_spec.values())[0][0]

        assert depth < len(nfeatures)
        self.depth = depth

        # Input feature embedding without batchnorm.
        self.embed_in = Conv1(D_in, embed_nin, embed_ks, st=(1,1,1))
        D_in = embed_nin

        # Contracting/downsampling pathway.
        for d in range(depth):
            fs = nfeatures[d]
            ks = sizes[d]
            self.add_conv_mod(d, D_in, fs, ks)
            self.add_max_pool(d+1, fs)
            D_in = fs

        # Bridge.
        fs = nfeatures[depth]
        ks = sizes[depth]
        self.add_conv_mod(depth, D_in, fs, ks)
        D_in = fs

        # Expanding/upsampling pathway.
        for d in reversed(range(depth)):
            fs = nfeatures[d]
            ks = sizes[d]
            self.add_upsample_mod(d, D_in, fs)
            D_in = fs
            self.add_dconv_mod(d, D_in, fs, ks)

        # Output feature embedding without batchnorm.
        self.embed_out = Conv1(D_in, embed_nout, embed_ks, st=(1,1,1))
        D_in = embed_nout

        # Output by spec.
        self.output = OutputMod(D_in, out_spec)

    def add_conv_mod(self, depth, D_in, D_out, ks):
        setattr(self, "convmod{}".format(depth), ConvMod(D_in,D_out,ks))

    def add_dconv_mod(self, depth, D_in, D_out, ks):
        setattr(self, "dconvmod{}".format(depth), ConvMod(D_in,D_out,ks))

    def add_max_pool(self, depth, D_in, down=(1,2,2)):
        setattr(self, "maxpool{}".format(depth), nn.MaxPool3d(down))

    def add_upsample_mod(self, depth, D_in, D_out, up=(1,2,2)):
        setattr(self, "upsample{}".format(depth), UpsampleMod(D_in,D_out,up))

    def forward(self, x):
        # Input feature embedding without batchnorm.
        x = self.embed_in(x)

        # Contracting/downsmapling pathway.
        skip = []
        for d in range(self.depth):
            convmod = getattr(self, "convmod{}".format(d))
            maxpool = getattr(self, "maxpool{}".format(d+1))
            x = convmod(x)
            skip.append(x)
            x = maxpool(x)

        # Bridge.
        bridge = getattr(self, "convmod{}".format(self.depth))
        x = bridge(x)

        # Expanding/upsampling pathway.
        for d in reversed(range(self.depth)):
            upsample = getattr(self, "upsample{}".format(d))
            dconvmod = getattr(self, "dconvmod{}".format(d))
            x = dconvmod(upsample(x, skip[d]))

        # Output feature embedding without batchnorm.
        x = self.embed_out(x)
        return self.output(x)
