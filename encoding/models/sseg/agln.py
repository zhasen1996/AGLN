###########################################################################
# Created by: Sen Zha
# Email: zha13051506858@gmail.com
# Copyright (c) 2022
###########################################################################

"""
AGLN: use SAB_SDM(A) and LRM.
"""

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

from .base import BaseNet

torch_ver = torch.__version__[:3]

__all__ = ['AGLN', 'get_agln', 'get_agln_50_ade']

class DepthwiseConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthwiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch,
                                    bias=False)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    bias=False)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class AGLN(BaseNet):
    r"""Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;

    Examples
    --------
    >>> model = AGLN(nclass=21, backbone='resnet50')
    >>> print(model)
    """

    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(AGLN, self).__init__(nclass, backbone, aux, se_loss, dilated=False, norm_layer=norm_layer)
        self.head = AGLNHead(nclass, norm_layer, up_kwargs=self._up_kwargs)
        assert not aux, "AGLN does not support aux loss"

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)  # e1, e2, e3, e4 from layer1, layer2, layer3, layer4

        x = list(self.head(*features))
        x[0] = interpolate(x[0], imsize, **self._up_kwargs)
        return tuple(x)


class AGLNHead(nn.Module):
    """
    encoder-decoder architecture:
    input                   output
         e1 ------------- d1
           e2 --------- d2
             e3 ----- d3
               e4 - d4
    """
    def __init__(self, out_channels, norm_layer=None, fpn_inchannels=[256, 512, 1024, 2048],
                 fpn_dim=256, up_kwargs=None):
        super(AGLNHead, self).__init__()
        # bilinear interpolate options
        assert up_kwargs is not None
        self._up_kwargs = up_kwargs
        fpn_lateral = []
        for fpn_inchannel in fpn_inchannels[:-1]:
            fpn_lateral.append(nn.Sequential(
                nn.Conv2d(fpn_inchannel, fpn_dim, kernel_size=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_lateral = nn.ModuleList(fpn_lateral)
        fpn_out = []
        for _ in range(len(fpn_inchannels) - 1):
            fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_out = nn.ModuleList(fpn_out)
        self.e4conv = nn.Sequential(nn.Conv2d(fpn_inchannels[-1], fpn_dim, 3, padding=1, bias=False),
                                    norm_layer(fpn_dim),
                                    nn.ReLU())
        inter_channels = len(fpn_inchannels) * fpn_dim
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))

        self.sab_d4 = SAB(fpn_dim, fpn_dim, fpn_dim // 4)
        self.sab_d3 = SAB(fpn_dim, fpn_dim, fpn_dim // 4)
        self.sab_d2 = SAB(fpn_dim, fpn_dim, fpn_dim // 4)

        self.cfb_d4 = nn.ModuleList(  # aggregate from d4, distribute to d3, d2, d1
            [CFB(fpn_dim, fpn_dim // 4, norm_layer=norm_layer),
            CFB(fpn_dim, fpn_dim // 4, norm_layer=norm_layer),
            CFB(fpn_dim, fpn_dim // 4, norm_layer=norm_layer)]
        )
        self.cfb_d3 = nn.ModuleList(  # aggregate from d3, distribute to d2, d1
            [CFB(fpn_dim, fpn_dim // 4, norm_layer=norm_layer),
             CFB(fpn_dim, fpn_dim // 4, norm_layer=norm_layer),
             Identity()]
        )
        self.cfb_d2 = nn.ModuleList(  # aggregate from d2, distribute to d1
            [CFB(fpn_dim, fpn_dim // 4, norm_layer=norm_layer),
             Identity(),
             Identity()]
        )

    def forward(self, *inputs):
        e4 = inputs[-1]
        feat = self.e4conv(e4)  # feat_e4: final output of encoder

        e1_size = inputs[0].size()[2:]
        feat_max = interpolate(feat, e1_size, **self._up_kwargs)
        fpn_features = [feat_max]

        # Initialization of the global semantic descriptors
        descriptors = [0, 0, 0]

        for i in reversed(range(len(inputs) - 1)):
            feat_i = self.fpn_lateral[i](inputs[i])  # encoder_feature: e4, 43, e2
            feat_up = interpolate(feat, feat_i.size()[2:], **self._up_kwargs)  # decoder_feature: d4, d3, d2

            # SABs
            if i == 2:
                descriptors[i] = self.sab_d4(feat)
            if i == 1:
                descriptors[i] = self.sab_d3(feat)
            if i == 0:
                descriptors[i] = self.sab_d2(feat)

            # CFBs
            feat_up = self.cfb_d4[i](feat_i, feat_up, descriptors[2])
            feat_up = self.cfb_d3[i](feat_i, feat_up, descriptors[1])
            feat = self.cfb_d2[i](feat_i, feat_up, descriptors[0])

            # interpolate to e1 size
            feat_max = interpolate(self.fpn_out[i](feat), e1_size, **self._up_kwargs)
            fpn_features.append(feat_max)
        fpn_features = torch.cat(fpn_features, 1)

        return (self.conv5(fpn_features),)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input_en, input_de, descriptors):
        return input_de


class SAB(nn.Module):
    """
    Semantic Aggregation Block:
        Aggregate global semantic descriptors from the encoder output.
    Params:
        c_in: input channels, same as fpn_dim(256)
        c_feat: feature channels, C in the paper (default as 256).
        c_atten: number of semantic descriptors, N in the paper (1, 64, 128, 256).
    """

    def __init__(self, c_in, c_feat, c_atten):
        super(SAB, self).__init__()
        self.c_feat = c_feat
        self.c_atten = c_atten
        self.conv_feat = nn.Conv2d(c_in, c_feat, kernel_size=1)
        self.conv_atten = nn.Conv2d(c_in, c_atten, kernel_size=1)

    def forward(self, input: torch.Tensor):
        b, c, h, w = input.size()
        feat = self.conv_feat(input).view(b, self.c_feat, -1)  # feature map
        atten = self.conv_atten(input).view(b, self.c_atten, -1)  # attention map
        atten = F.softmax(atten, dim=-1)
        descriptors = torch.bmm(feat, atten.permute(0, 2, 1))  # (c_feat, c_atten)

        return descriptors


class SDM(nn.Module):
    """
    Semantic Distribution Module:
        Distribute global semantic descriptors to each stage of decoder.
    Params:
        c_atten: number of semantic descriptors, N in the paper.
        c_de: decoder channels
    """

    def __init__(self, c_atten, c_de):
        super(SDM, self).__init__()
        self.c_atten = c_atten
        self.conv_de = nn.Conv2d(c_de, c_atten, kernel_size=1)
        self.out_conv = nn.Conv2d(c_de, c_de, kernel_size=1)

    def forward(self, descriptors: torch.Tensor, input_de: torch.Tensor):
        b, c, h, w = input_de.size()
        atten_vectors = F.softmax(self.conv_de(input_de), dim=1)
        output = descriptors.matmul(atten_vectors.view(b, self.c_atten, -1)).view(b, -1, h, w)

        return self.out_conv(output)


class LRM(nn.Module):
    """
    Local Refinement Module: including channel resampling and spatial gating.
    Params:
        c_en: encoder channels
        c_de: decoder channels
    """
    def __init__(self, c_en, c_de):
        super(LRM, self).__init__()
        self.c_en = c_en
        self.c_de = c_de

    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, gate_map):
        b, c, h, w = input_de.size()
        input_en = input_en.view(b, self.c_en, -1)

        # Channel Resampling
        energy = input_de.view(b, self.c_de, -1).matmul(input_en.transpose(-1, -2))
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(
            energy) - energy  # Prevent loss divergence during training
        channel_attention_map = torch.softmax(energy_new, dim=-1)
        input_en = channel_attention_map.matmul(input_en).view(b, -1, h, w)  # channel_attention_feat

        # Spatial Gating
        gate_map = torch.sigmoid(gate_map)
        input_en = input_en.mul(gate_map)

        return input_en


class CFB(nn.Module):
    """
    Context Fusion Block: including SDM and LRM.
    Params:
        c_atten: number of semantic descriptors, N in the paper.
    """

    def __init__(self, fpn_dim=256, c_atten=256, norm_layer=None, ):
        super(CFB, self).__init__()
        self.sdm = SDM(c_atten, fpn_dim)
        self.lrm = LRM(fpn_dim, fpn_dim)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            # nn.Conv2d(3 * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            # DepthwiseConv(3 * fpn_dim, fpn_dim),
            norm_layer(fpn_dim),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, global_descripitors: torch.Tensor):
        feat_global = self.sdm(global_descripitors, input_de)
        feat_local = self.gamma * self.lrm(input_en, input_de, feat_global) + input_en
        # add fusion
        return self.conv_fusion(input_de + self.alpha * feat_global + self.beta * feat_local)
        # concat fusion
        # return self.conv_fusion(torch.cat((input_de, self.beta * feat_global, self.gamma * feat_local), dim=1))


def get_agln(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                  root='~/.encoding/models', **kwargs):
    r"""
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_agln(dataset='pascal_voc', backbone='resnet50s', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = AGLN(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('agln_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model


def get_agln_50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""
    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_agln_50_ade(pretrained=True)
    >>> print(model)
    """
    return get_agln('ade20k', 'resnet50s', pretrained)
