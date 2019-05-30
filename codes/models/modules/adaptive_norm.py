import math

import torch.nn as nn
import torch
import torch.nn.functional as F


class AdaptiveConvResNorm(nn.Module):
    """
    norm the output with scale and shift without substrat the mean and variance
    """

    def __init__(self, in_channel, kernel_size):

        super(AdaptiveConvResNorm, self).__init__()
        padding = _compute_padding(kernel_size)
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding,
                                     groups=in_channel, bias=True)

    def forward(self, x):
        return self.transformer(x) + x


def _compute_padding(kernel_size):
    # compute the size of padding
    return (kernel_size - 1) // 2


class GateNonLinearLayer(nn.Module):
    """
    control the scale and shift with a single value
    """
    def __init__(self, in_nc, conv_bias):
        super(GateNonLinearLayer, self).__init__()
        self.in_nc = in_nc
        self.conv_bias = conv_bias
        self.mid_filters = 32
        self.Gate_scale_conv0 = nn.Conv2d(in_nc, self.mid_filters, 3, bias=conv_bias, stride=1, padding=1)
        self.Gate_scale_conv1 = nn.Conv2d(self.mid_filters, 64, 1, bias=conv_bias)
        self.Gate_shift_conv0 = nn.Conv2d(in_nc, self.mid_filters, 3, bias=conv_bias, stride=1, padding=1)
        self.Gate_shift_conv1 = nn.Conv2d(self.mid_filters, 64, 1, bias=conv_bias)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        # x = [i*1.01 for i in x]
        scale = self.Gate_scale_conv1(F.leaky_relu(self.Gate_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.Gate_shift_conv1(F.leaky_relu(self.Gate_shift_conv0(x[1]), 0.1, inplace=True))
        # scale = F.sigmoid(self.Gate_scale_conv1(F.leaky_relu(self.Gate_scale_conv0(x[1]), 0.1, inplace=True)))
        # shift = F.sigmoid(self.Gate_shift_conv1(F.leaky_relu(self.Gate_shift_conv0(x[1]), 0.1, inplace=True)))
        return x[0] * (scale + 1) + shift

    def __repr__(self):
        return "GateNonLinearLayer({}, conv_bias={})".format(self.in_nc, self.conv_bias)


class MetaLayer(nn.Module):
    """
    a subnet that can control the base network
    """

    def __init__(self, in_nc, conv_bias, kernel_size):
        super(MetaLayer, self).__init__()
        self.in_nc = in_nc
        self.kernel_size = kernel_size
        self.padding = _compute_padding(kernel_size)
        self.conv_bias = conv_bias

        # self.Gate_scale_conv0 = nn.Conv2d(1, in_nc, 1, bias=conv_bias)
        # self.Gate_shift_conv0 = nn.Conv2d(1, in_nc, 1, bias=conv_bias)

        self.Gate_scale_conv0 = nn.Conv2d(1, 32, 1, bias=conv_bias)
        self.Gate_scale_conv1 = nn.Conv2d(32, in_nc, 1, bias=conv_bias)
        self.Gate_shift_conv0 = nn.Conv2d(1, 32, 1, bias=conv_bias)
        self.Gate_shift_conv1 = nn.Conv2d(32, in_nc, 1, bias=conv_bias)

    def forward(self, x):
        scale_input = x[1][0]
        shift_input = x[1][1]
        # scale = self.Gate_scale_conv0(scale_input)
        # shift = self.Gate_shift_conv0(shift_input)
        scale = self.Gate_scale_conv1(F.leaky_relu(self.Gate_scale_conv0(scale_input), 0.1, inplace=True))
        shift = self.Gate_shift_conv1(F.leaky_relu(self.Gate_shift_conv0(shift_input), 0.1, inplace=True))
        scale_reshape = scale.view(64, 1, 3, 3)
        shift_reshape = shift.view(-1)
        return F.conv2d(x[0], scale_reshape, shift_reshape, 1, self.padding, 1, self.in_nc) + x[0]

    def __repr__(self):
        return "MetaLayer(in_channel={}, out_channel={}, kernel_size={}, padding={}, " \
               "conv_bias={})".format(self.in_nc, self.in_nc, self.kernel_size, self.padding, self.conv_bias)


class InterpAdaptiveResNorm(nn.Module):
    """
    norm the output with scale and shift without substrat the mean and variance
    """

    def __init__(self, in_channel, kernel_size):

        super(InterpAdaptiveResNorm, self).__init__()
        padding = _compute_padding(kernel_size)
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size, padding=padding,
                                     groups=in_channel, bias=True)

    def forward(self, x, y):
        return self.transformer(x)*y + x


class CondAdaptiveConvResNorm(nn.Module):

    def __init__(self, num_features, num_classes):
        super(CondAdaptiveConvResNorm, self).__init__()
        self.num_features = num_features
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].uniform_()
        self.embed.weight.data[:, num_features:].zero_()
        # self.embed.weight.data[:, num_features:].uniform_()

    def forward(self, x, y):
        gamma, beta = self.embed(y).chunk(2, -1)
        out = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1, self.num_features, 1, 1)
        return out + x


class CondResTransformer(nn.Module):

    def __init__(self, num_features, kernel_size, num_classes):
        super(CondResTransformer, self).__init__()
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.padding = _compute_padding(kernel_size)
        self.embed = nn.Embedding(num_classes, (kernel_size**2+1)*num_features)
        self.reset_parameters()
        # self.embed.weight.data[:, :-num_features].uniform_()
        # self.embed.weight.data[:, -num_features:].zero_()
        # self.embed.weight.data[:, num_features:].uniform_()

    def reset_parameters(self):
        n = self.num_features
        k = self.kernel_size
        n *= k*k
        stdv = 1. / math.sqrt(n)
        self.embed.weight.data[:, :-self.num_features].uniform_(-stdv, stdv)
        self.embed.weight.data[:, -self.num_features:].uniform_(-stdv, stdv)

    def forward(self, x, y):
        params = self.embed(y).squeeze()
        weight = params[:-self.num_features].view(self.num_features, 1, self.kernel_size, self.kernel_size)
        bias = params[-self.num_features:]
        return F.conv2d(x, weight, bias, 1, self.padding, 1, self.num_features) + x

# class CondAdaptiveConvResNorm(nn.Module):
#     """
#     norm the output with scale and shift without substrat the mean and variance
#     """
#
#     def __init__(self, num_features, num_classes):
#         super(CondAdaptiveConvResNorm, self).__init__()
#         self.features = num_features
#         self.gamma = nn.Parameter(torch.zeros(1, num_features, 1, 1))
#         self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
#
#     def forward(self, x, y):
#         return (self.gamma + 1) * x + self.beta


class CondInstanceNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super(CondInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.insnorm = nn.InstanceNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.insnorm(x)
        gamma, beta = self.embed(y).chunk(2, -1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

