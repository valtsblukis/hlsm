import torch
from torch import nn as nn
import torch.nn.functional as F

from lgp.models.alfred.hlsm.unets.unet_blocks import UpscaleDoubleConv, DoubleConv, objectview


PROFILE = False


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

# TODO: This shouldn't live inside what should be a completely reusable module.
from lgp.models.alfred.hlsm.hlsm_state_repr import AlfredSpatialStateRepr


class Lingunet5(torch.nn.Module):
    def __init__(self):
        super(Lingunet5, self).__init__()

        params = {
            "in_channels": AlfredSpatialStateRepr.get_num_data_channels(),
            "hc1": 32,
            "hb1": 16,
            "hc2": 32,
            "out_channels": 1,
            "embedding_size": 32,
            "stride": 2
        }

        self.p = objectview(params)
        self.emb_block_size = self.p.embedding_size

        DeconvOp = UpscaleDoubleConv
        ConvOp = DoubleConv

        # inchannels, outchannels, kernel size
        self.conv1 = ConvOp(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv1 = DeconvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv5 = DeconvOp(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)

        self.fnorm1 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm2 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm3 = nn.InstanceNorm2d(self.p.hb1)
        self.fnorm4 = nn.InstanceNorm2d(self.p.hb1)

        self.lang19 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang28 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang37 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang46 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hb1)
        self.lang55 = nn.Linear(self.emb_block_size, self.p.hc1 * self.p.hc1)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()
        #self.deconv5.init_weights()

    def forward(self, input, ctx, tensor_store=None):
        ctx = ctx.contiguous()
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))
        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.act(self.conv5(x4))

        bs = input.shape[0]

        if ctx is not None:
            ctx = F.normalize(ctx, p=2, dim=1)

            lf1 = F.normalize(self.lang19(ctx)).view([bs, self.p.hb1, self.p.hc1])
            lf2 = F.normalize(self.lang28(ctx)).view([bs, self.p.hb1, self.p.hc1])
            lf3 = F.normalize(self.lang37(ctx)).view([bs, self.p.hb1, self.p.hc1])
            lf4 = F.normalize(self.lang46(ctx)).view([bs, self.p.hb1, self.p.hc1])
            lf5 = F.normalize(self.lang55(ctx)).view([bs, self.p.hc1, self.p.hc1])

            x1f = torch.einsum("bchw,bdc->bdhw", x1, lf1)
            x2f = torch.einsum("bchw,bdc->bdhw", x2, lf2)
            x3f = torch.einsum("bchw,bdc->bdhw", x3, lf3)
            x4f = torch.einsum("bchw,bdc->bdhw", x4, lf4)
            x5f = torch.einsum("bchw,bdc->bdhw", x5, lf5)

            x1 = self.fnorm1(x1f)
            x2 = self.fnorm2(x2f)
            x3 = self.fnorm3(x3f)
            x4 = self.fnorm4(x4f)
            x5 = x5f

        x6 = self.act(self.deconv1(x5, output_size=x4.size()))
        x46 = torch.cat([x4, x6], 1)
        x7 = self.dnorm3(self.act(self.deconv2(x46, output_size=x3.size())))
        x37 = torch.cat([x3, x7], 1)
        x8 = self.dnorm4(self.act(self.deconv3(x37, output_size=x2.size())))
        x28 = torch.cat([x2, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1, x9], 1)
        map_scores = self.deconv5(x19, output_size=input.size())

        return map_scores
