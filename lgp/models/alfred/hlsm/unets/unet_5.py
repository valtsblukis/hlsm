import torch
from torch import nn as nn
import torch.nn.functional as F

from lgp.models.alfred.hlsm.unets.unet_blocks import UpscaleDoubleConv, DoubleConv, objectview
from lgp.ops.spatial_distr import multidim_logsoftmax

import lgp.env.alfred.segmentation_definitions as segdef


class UNet5(torch.nn.Module):
    def __init__(self, distr_depth, depth_bins):
        super(UNet5, self).__init__()

        self.num_c = segdef.get_num_objects()
        self.depth_bins = depth_bins

        params = {
            "in_channels": 3,
            "hc1": 256,
            "hc2": 256,
            "out_channels": self.num_c + self.depth_bins if distr_depth else self.num_c + 1,
            "out_vec_length": self.num_c + 1,
            "stride": 2
        }

        self.p = objectview(params)
        self.distr_depth = distr_depth

        DeconvOp = UpscaleDoubleConv
        ConvOp = DoubleConv

        # inchannels, outchannels, kernel size
        self.conv1 = ConvOp(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv6 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv1 = DeconvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv5 = DeconvOp(self.p.hc1 * 2, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv6 = DeconvOp(self.p.hc1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        self.norm6 = nn.InstanceNorm2d(self.p.hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)

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

    def forward(self, input):
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))

        x3 = self.dropout(x3)

        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.norm6(self.act(self.conv5(x4)))
        x6 = self.act(self.conv6(x5))

        x6 = self.dropout(x6)

        y5 = self.act(self.deconv1(x6, output_size=x5.size()))
        xy5 = torch.cat([x5, y5], 1)

        y4 = self.dnorm3(self.act(self.deconv2(xy5, output_size=x4.size())))
        xy4 = torch.cat([x4, y4], 1)
        y3 = self.dnorm4(self.act(self.deconv3(xy4, output_size=x3.size())))
        xy3 = torch.cat([x3, y3], 1)
        y2 = self.dnorm4(self.act(self.deconv4(xy3, output_size=x2.size())))
        xy2 = torch.cat([x2, y2], 1)

        xy2 = self.dropout(xy2)

        y1 = self.dnorm5(self.act(self.deconv5(xy2, output_size=x1.size())))
        xy1 = torch.cat([x1, y1], 1)
        out = self.deconv6(xy1, output_size=input.size())

        out_a = out[:, :self.num_c]
        out_b = out[:, self.num_c:]

        out_a = multidim_logsoftmax(out_a, dims=(1,))

        if self.distr_depth:
            out_b = multidim_logsoftmax(out_b, dims=(1,))

        return out_a, out_b