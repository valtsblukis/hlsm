import torch
from torch import nn as nn
import torch.nn.functional as F

from lgp.models.alfred.hlsm.unets.unet_blocks import UpscaleDoubleConv, DoubleConv, objectview

PROFILE = False


class Lingunet3(torch.nn.Module):
    def __init__(self, in_channels, context_size, out_channels):
        super(Lingunet3, self).__init__()

        params = {
            "in_channels": in_channels,
            "hc1": 32,
            "hb1": 16,
            "hc2": 32,
            "out_channels": out_channels,
            "context_size": context_size,
            "stride": 2
        }
        self.p = objectview(params)

        self.emb_block_size = self.p.context_size

        DeconvOp = UpscaleDoubleConv
        ConvOp = DoubleConv

        # inchannels, outchannels, kernel size
        self.conv1 = ConvOp(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv3 = DeconvOp(self.p.hb1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc1 + self.p.hb1, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv5 = DeconvOp(self.p.hb1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)


        self.act = nn.LeakyReLU()
        self.input_dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
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

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()

    def forward(self, inp, ctx):
        ctx = ctx.contiguous()

        inp = self.input_dropout(inp)

        x1 = self.norm2(self.act(self.conv1(inp)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))

        x3 = self.dropout2(x3)

        bs = inp.shape[0]

        if ctx is not None:
            ctx = F.normalize(ctx, p=2, dim=1)

            lf1 = F.normalize(self.lang19(ctx)).view([bs, self.p.hb1, self.p.hc1])
            lf2 = F.normalize(self.lang28(ctx)).view([bs, self.p.hb1, self.p.hc1])
            lf3 = F.normalize(self.lang37(ctx)).view([bs, self.p.hb1, self.p.hc1])

            x1f = torch.einsum("bchw,bdc->bdhw", x1, lf1)
            x2f = torch.einsum("bchw,bdc->bdhw", x2, lf2)
            x3f = torch.einsum("bchw,bdc->bdhw", x3, lf3)

            x1 = self.fnorm1(x1f)
            x2 = self.fnorm2(x2f)
            x3 = self.fnorm3(x3f)

        x8 = self.dnorm4(self.act(self.deconv3(x3, output_size=x2.size())))
        x28 = torch.cat([x2, x8], 1)
        x9 = self.dnorm5(self.act(self.deconv4(x28, output_size=x1.size())))
        x19 = torch.cat([x1, x9], 1)
        map_scores = self.deconv5(x19, output_size=inp.size())

        return map_scores
