from collections import OrderedDict
import torch
import torch.nn as nn
import pdb

from pytorch_wavelets import DWTForward, DWTInverse 

####################
# Basic blocks
####################


def act(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ConcatBlock(nn.Module):
    # Concat the output of a submodule to its input
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)


####################
# Useful blocks
####################


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        # if in_nc != out_nc:
        #     self.project = conv_block(in_nc, out_nc, 1, stride, dilation, 1, bias, pad_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


class ResidualDenseBlock_5C(nn.Module):
    '''
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x

class ResidualDenseBlockTiny_4C(nn.Module):
    '''
    Residual Dense Block
    style: 4 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=16, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(ResidualDenseBlockTiny_4C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv4 = conv_block(nc+3*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4.mul(0.2) + x

class RRDBTiny(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=16, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDBTiny, self).__init__()
        self.RDB1 = ResidualDenseBlockTiny_4C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlockTiny_4C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        return out.mul(0.2) + x

class RRDB(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x

class MF8T4DB(nn.Module):
    '''
    input 8 LR frames into network, but we fisrt average neighboring 2 frames into 1 frame, \
    so we indeed take 4 frames as input.
    '''
    def __init__(self, in_nc, out_nc=4, stride=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='relu'):
        super(MF8T4DB, self).__init__()
        self.conv1_ref = nn.Conv2d(in_nc, out_nc, 5, padding=2, stride=1)
        self.relu1_ref = nn.PReLU(num_parameters=out_nc)
        self.conv1 = nn.Conv2d(in_nc, out_nc, 5, padding=2, stride=1)
        self.relu1 = nn.PReLU(num_parameters=out_nc)

    def forward(self, x):
        x = x * 0.5
        N,C,H,W = x.size()
        assert C == 8
        split_dim = 1
        split_x = torch.split(tensor=x, split_size_or_sections=2, dim=split_dim)
        split_sum = torch.sum(split_x[0], dim=split_dim, keepdim=True)
        conv1_x = []
        conv1_x.append(self.relu1_ref(self.conv1_ref(split_sum)))
        for i in range(1, 4):
            split_sum = torch.sum(split_x[i], dim = split_dim, keepdim=True)
            conv1_x.append(self.relu1(self.conv1(split_sum)))
        x = torch.cat(conv1_x, dim=split_dim)
        return x

class MF8IMGDB(nn.Module):
    '''
    input 8 LR frames into network
    '''
    def __init__(self, in_nc, out_nc=4, stride=1, bias=True, pad_type='zero', \
                 norm_type=None, act_type='relu'):
        super(MF8IMGDB, self).__init__()
        self.conv1_ref = nn.Conv2d(in_nc, out_nc, 5, padding=2, stride=1)
        self.relu1_ref = nn.PReLU(num_parameters=out_nc)
        self.conv1 = nn.Conv2d(in_nc, out_nc, 5, padding=2, stride=1)
        self.relu1 = nn.PReLU(num_parameters=out_nc)

    def forward(self, x):
        N,C,H,W = x.size()
        assert C == 8
        split_dim = 1
        conv1_x = []
        conv1_x.append(self.relu1_ref(self.conv1_ref(x[:, 0, :, :].unsqueeze(1))))
        for i in range(1, 8):
            conv1_x.append(self.relu1(self.conv1(x[:, i, :, :].unsqueeze(1))))
        x = torch.cat(conv1_x, dim=split_dim)

        return x

class Res_Module(nn.Module):
    def __init__(self):
        super(Res_Module,self).__init__()
        self.conv_seg = nn.Conv2d(1,1,5,padding=2,bias=False,stride=2)
        self.single_conv1 = nn.Conv2d(8,8,3,padding=1,bias=False)

    def forward(self,x,mask):
        mask = self.conv_seg(mask)
        mask = mask.expand(mask.size()[0], 8, mask.size()[2], mask.size()[3])
        x_res = self.single_conv1(x*mask)
        x = x + x_res

        return x

class Res_Module_ch16_nods(nn.Module):
    def __init__(self):
        super(Res_Module_ch16_nods, self).__init__()
        self.conv_seg = nn.Conv2d(1,1,5,padding=2,bias=False,stride=1)
        self.single_conv1 = nn.Conv2d(16,16,3,padding=1,bias=False)

    def forward(self,x,mask):
        mask = self.conv_seg(mask)
        mask = mask.expand(mask.size()[0], 16, mask.size()[2], mask.size()[3])
        x_res = self.single_conv1(x*mask)
        x = x + x_res

        return x

class Res_Module_ch16(nn.Module):
    def __init__(self):
        super(Res_Module_ch16, self).__init__()
        self.conv_seg = nn.Conv2d(1,1,5,padding=2,bias=False,stride=1)
        self.single_conv1 = nn.Conv2d(16,16,3,padding=1,bias=False)

    def forward(self,x,mask):
        mask = self.conv_seg(mask)
        mask = mask.expand(mask.size()[0], 16, mask.size()[2], mask.size()[3])
        x_res = self.single_conv1(x*mask)
        x = x + x_res

        return x

class Res_Module_ch32_nods(nn.Module):
    def __init__(self):
        super(Res_Module_ch32_nods, self).__init__()
        self.conv_seg = nn.Conv2d(1,1,5,padding=2,bias=False,stride=1)
        self.single_conv1 = nn.Conv2d(32,32,3,padding=1,bias=False)

    def forward(self,x,mask):
        mask = self.conv_seg(mask)
        mask = mask.expand(mask.size()[0], 32, mask.size()[2], mask.size()[3])
        x_res = self.single_conv1(x*mask)
        x = x + x_res

        return x

class Res_Module_ch32(nn.Module):
    def __init__(self):
        super(Res_Module_ch32, self).__init__()
        self.conv_seg = nn.Conv2d(1,1,5,padding=2,bias=False,stride=2)
        self.single_conv1 = nn.Conv2d(32,32,3,padding=1,bias=False)

    def forward(self,x,mask):
        mask = self.conv_seg(mask)
        mask = mask.expand(mask.size()[0], 32, mask.size()[2], mask.size()[3])
        x_res = self.single_conv1(x*mask)
        x = x + x_res

        return x

class Res_Module_ch64(nn.Module):
    def __init__(self):
        super(Res_Module_ch64, self).__init__()
        self.conv_seg = nn.Conv2d(1,1,5,padding=2,bias=False,stride=2)
        self.single_conv1 = nn.Conv2d(64,64,3,padding=1,bias=False)

    def forward(self,x,mask):
        mask = self.conv_seg(mask)
        mask = mask.expand(mask.size()[0], 64, mask.size()[2], mask.size()[3])
        x_res = self.single_conv1(x*mask)
        x = x + x_res

        return x
####################
# Upsampler
####################
def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, act_type='relu'):
    '''
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    '''
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, act_type=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, act_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, act_type=act_type)
    return sequential(upsample, conv)

class CascadeBlock(nn.Module):
    """
    CascadeBlock, 3-3 style
    """

    def __init__(self, nc, gc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(CascadeBlock, self).__init__()
        self.nc = nc
        self.ResBlocks = nn.ModuleList([ResNetBlock(gc, gc, gc, kernel_size, stride, dilation, groups, bias, \
            pad_type, norm_type, act_type, mode, res_scale) for _ in range(nc)])
        self.CatBlocks = nn.ModuleList([conv_block((i + 2)*gc, gc, kernel_size=1, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nc)])

    def forward(self, x):
        pre_fea = x
        for i in range(self.nc):
            res = self.ResBlocks[i](x)
            pre_fea = torch.cat((pre_fea, res), dim=1)
            x = self.CatBlocks[i](pre_fea)
        return x

class CCAB(nn.Module):
    """
    Cascade Channel Attention Block, 3-3 style
    """

    def __init__(self, nc, gc, kernel_size=3, stride=1, dilation=1, groups=1, reduction=16, \
            bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(CCAB, self).__init__()
        self.nc = nc
        self.RCAB = nn.ModuleList([RCAB(gc, kernel_size, reduction, stride, dilation, groups, bias, pad_type, \
                    norm_type, act_type, mode, res_scale) for _ in range(nc)])
        self.CatBlocks = nn.ModuleList([conv_block((i + 2)*gc, gc, kernel_size=1, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nc)])

    def forward(self, x):
        pre_fea = x
        for i in range(self.nc):
            res = self.RCAB[i](x)
            pre_fea = torch.cat((pre_fea, res), dim=1)
            x = self.CatBlocks[i](pre_fea)
        return x

class RCAB(nn.Module):
    ## Residual Channel Attention Block (RCAB)
    def __init__(self, nf, kernel_size=3, reduction=16, stride=1, dilation=1, groups=1, bias=True, \
            pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(RCAB, self).__init__()
        self.res = sequential(
            conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, pad_type, \
                        norm_type, act_type, mode),
            conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, pad_type, \
                        norm_type, None, mode),
            CALayer(nf, reduction, stride, dilation, groups, bias, pad_type, norm_type, act_type, mode)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res

class CALayer(nn.Module):
    # Channel Attention (CA) Layer
    def __init__(self, channel, reduction=16, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.attention = sequential(
                nn.AdaptiveAvgPool2d(1),
                conv_block(channel, channel // reduction, 1, stride, dilation, groups, bias, pad_type, \
                            norm_type, act_type, mode),
                conv_block(channel // reduction, channel, 1, stride, dilation, groups, bias, pad_type, \
                            norm_type, None, mode),
                nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)

class ResidualGroupBlock(nn.Module):
    ## Residual Group (RG)
    def __init__(self, nf, nb, kernel_size=3, reduction=16, stride=1, dilation=1, groups=1, bias=True, \
            pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1):
        super(ResidualGroupBlock, self).__init__()
        group = [RCAB(nf, kernel_size, reduction, stride, dilation, groups, bias, pad_type, \
                    norm_type, act_type, mode, res_scale) for _ in range(nb)]
        conv = conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, pad_type, \
                        norm_type, None, mode)
        self.res = sequential(*group, conv)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res

class pixel_shuffle_ds(nn.Module):
    def __init__(self, ds_factor):
        super(pixel_shuffle_ds, self).__init__()
        self.ds_factor = ds_factor
    def forward(self, x):
        n, c, h, w = x.size()
        assert h % self.ds_factor == 0 and w % self.ds_factor == 0
        x = x.permute((0, 2, 3, 1))
        x = x.unfold(1, self.ds_factor, self.ds_factor).unfold(2, self.ds_factor, self.ds_factor).contiguous()
        x = x.view(n, h / self.ds_factor, w / self.ds_factor, -1).permute((0, 3, 1, 2))
        return x


####################
# Block for DBPN
####################
class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, \
                activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = conv_block(num_filter, num_filter, kernel_size, stride, act_type=activation, norm_type=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)        

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, \
                activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = conv_block(num_filter, num_filter, kernel_size, stride, act_type=activation, norm_type=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = conv_block(num_filter, num_filter, kernel_size, stride, act_type=activation, norm_type=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, \
                bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

####################
# Block for OctConv
####################
class OctaveConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
                    bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
        super(OctaveConv, self).__init__()
        assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
        padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

        self.l2l = nn.Conv2d(int(alpha * in_nc), int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = nn.Conv2d(int(alpha * in_nc), out_nc - int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = nn.Conv2d(in_nc - int(alpha * in_nc), int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_nc - int(alpha * in_nc), out_nc - int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.a = act(act_type) if act_type else None
        self.n_h = norm(norm_type, int(out_nc*(1 - alpha))) if norm_type else None
        self.n_l = norm(norm_type, int(out_nc*alpha)) if norm_type else None

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(self.l2h(X_l))
        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(self.h2g_pool(X_h))
        
        #print(X_l2h.shape,"~~~~",X_h2h.shape)
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        if self.n_h and self.n_l:
            X_h = self.n_h(X_h)
            X_l = self.n_l(X_l)

        if self.a:
            X_h = self.a(X_h)
            X_l = self.a(X_l)

        return X_h, X_l


class FirstOctaveConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
                    bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
        super(FirstOctaveConv, self).__init__()
        assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
        padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.stride = stride
        self.h2l = nn.Conv2d(in_nc, int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_nc, out_nc - int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.a = act(act_type) if act_type else None
        self.n_h = norm(norm_type, int(out_nc*(1 - alpha))) if norm_type else None
        self.n_l = norm(norm_type, int(out_nc*alpha)) if norm_type else None

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h = self.h2h(x)
        X_l = self.h2l(self.h2g_pool(x))

        if self.n_h and self.n_l:
            X_h = self.n_h(X_h)
            X_l = self.n_l(X_l)

        if self.a:
            X_h = self.a(X_h)
            X_l = self.a(X_l)

        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
                    bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
        super(LastOctaveConv, self).__init__()
        assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
        padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride

        self.l2h = nn.Conv2d(int(alpha * in_nc), out_nc,
                                kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_nc - int(alpha * in_nc), out_nc,
                                kernel_size, 1, padding, dilation, groups, bias)

        self.a = act(act_type) if act_type else None
        self.n_h = norm(norm_type, out_nc) if norm_type else None

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)
        
        X_h2h = self.h2h(X_h)
        X_l2h = self.upsample(self.l2h(X_l))
        
        X_h = X_h2h + X_l2h

        if self.n_h:
            X_h = self.n_h(X_h)

        if self.a:
            X_h = self.a(X_h)

        return X_h

class OctaveCascadeBlock(nn.Module):
    """
    OctaveCascadeBlock, 3-3 style
    """
    def __init__(self, nc, gc, kernel_size=3, alpha=0.75, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(OctaveCascadeBlock, self).__init__()
        self.nc = nc
        self.ResBlocks = nn.ModuleList([OctaveResBlock(gc, gc, gc, kernel_size, alpha, stride, dilation, \
            groups, bias, pad_type, norm_type, act_type, mode, res_scale) for _ in range(nc)])
        self.CatBlocks = nn.ModuleList([OctaveConv((i + 2)*gc, gc, kernel_size=1, alpha=alpha, bias=bias, \
            pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nc)])

    def forward(self, x):
        pre_fea = x
        for i in range(self.nc):
            res = self.ResBlocks[i](x)
            pre_fea = (torch.cat((pre_fea[0], res[0]), dim=1), \
                        torch.cat((pre_fea[1], res[1]), dim=1))
            x = self.CatBlocks[i](pre_fea)
        return x

class OctaveResBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, alpha=0.75, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(OctaveResBlock, self).__init__()
        conv0 = OctaveConv(in_nc, mid_nc, kernel_size, alpha, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)
        if mode == 'CNA':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = OctaveConv(mid_nc, out_nc, kernel_size, alpha, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode)

        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        #if(len(x)>2):
            #print(x[0].shape,"  ",x[1].shape,"  ",x[2].shape,"  ",x[3].shape)
        #print(len(x))
        res = self.res(x)
        res = (res[0].mul(self.res_scale), res[1].mul(self.res_scale))
        x = (x[0] + res[0], x[1] + res[1])
        #print(len(x),"~~~",len(res),"~~~",len(x + res))

        #return (x[0] + res[0], x[1]+res[1])
        return x

################################################################################################
class octave_ResidualDenseBlockTiny_4C(nn.Module):
    '''
    Residual Dense Block
    style: 4 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=16,alpha=0.5, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(octave_ResidualDenseBlockTiny_4C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 =OctaveConv(nc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
             norm_type=norm_type, act_type=act_type, mode=mode) 
        # conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = OctaveConv(nc+gc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
             norm_type=norm_type, act_type=act_type, mode=mode) 
        # conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = OctaveConv(nc+2*gc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
             norm_type=norm_type, act_type=act_type, mode=mode) 
        # conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv4 = OctaveConv(nc+3*gc, nc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
             norm_type=norm_type, act_type=act_type, mode=mode) 
        # conv_block(nc+3*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2((torch.cat((x[0], x1[0]), dim=1),(torch.cat((x[1], x1[1]), dim=1))))
        x3 = self.conv3((torch.cat((x[0], x1[0],x2[0]), dim=1),(torch.cat((x[1], x1[1],x2[1]), dim=1))))
        x4 = self.conv4((torch.cat((x[0], x1[0],x2[0],x3[0]), dim=1),(torch.cat((x[1], x1[1],x2[1],x3[1]), dim=1))))

        res = (x4[0].mul(0.2), x4[1].mul(0.2))
        x = (x[0] + res[0], x[1] + res[1])
        #print(len(x),"~~~",len(res),"~~~",len(x + res))

        #return (x[0] + res[0], x[1]+res[1])
        return x



####################################################################################################################
class octave_RRDBTiny(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''

    def __init__(self, nc, kernel_size=3, gc=16, stride=1, alpha=0.5, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(octave_RRDBTiny, self).__init__()
        self.RDB1 = octave_ResidualDenseBlockTiny_4C(nc=nc, kernel_size=kernel_size,alpha=alpha, gc=gc, stride=stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.RDB2 = octave_ResidualDenseBlockTiny_4C(nc=nc, kernel_size=kernel_size,alpha=alpha, gc=gc, stride=stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)

        res = (out[0].mul(0.2), out[1].mul(0.2))
        x = (x[0] + res[0], x[1] + res[1])
        #print(len(x),"~~~",len(res),"~~~",len(x + res))

        #return (x[0] + res[0], x[1]+res[1])
        return x

# ##################################################################################
# ##################################################################################
# ##################################################################################
# #modified octave
# # Block for OctConv
# ####################
# class M_NP_OctaveConv(nn.Module):
#     def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
#                     bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
#         super(M_NP_OctaveConv, self).__init__()
#         assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
#         padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
#         self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         self.h2g_pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsample = nn.Upsample(scale_factor=4, mode='nearest')#double pool
#         self.stride = stride

#         self.l2l = nn.Conv2d(int(alpha * in_nc), int(alpha * out_nc),
#                                 kernel_size, 1, padding, dilation, groups, bias)
#         self.l2h = nn.Conv2d(int(alpha * in_nc), out_nc - int(alpha * out_nc),
#                                 kernel_size, 1, padding, dilation, groups, bias)
#         self.h2l = nn.Conv2d(in_nc - int(alpha * in_nc), int(alpha * out_nc),
#                                 kernel_size, 1, padding, dilation, groups, bias)
#         self.h2h = nn.Conv2d(in_nc - int(alpha * in_nc), out_nc - int(alpha * out_nc),
#                                 kernel_size, 1, padding, dilation, groups, bias)
#         self.a = act(act_type) if act_type else None
#         self.n_h = norm(norm_type, int(out_nc*(1 - alpha))) if norm_type else None
#         self.n_l = norm(norm_type, int(out_nc*alpha)) if norm_type else None

#     def forward(self, x):
#         X_h, X_l = x

#         #if self.stride ==2:
#             #X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

#         X_h2h = self.h2h(X_h)
#         #X_l2h = self.upsample(self.l2h(X_l))
#         #X_l2h = self.l2h(X_l)
#         X_l2h = self.upsample(self.l2h(X_l))

#         X_l2l = self.l2l(X_l)
#         #X_h2l = self.h2l(self.h2g_pool(X_h))
#         #X_h2l = self.h2l(X_h)
#         X_h2l = self.h2l(self.h2g_pool2(self.h2g_pool(X_h)))
        
#         #print(X_l2h.shape,"~~~~",X_h2h.shape)
#         X_h = X_l2h + X_h2h
#         X_l = X_h2l + X_l2l

#         if self.n_h and self.n_l:
#             X_h = self.n_h(X_h)
#             X_l = self.n_l(X_l)

#         if self.a:
#             X_h = self.a(X_h)
#             X_l = self.a(X_l)

#         return X_h, X_l


# class M_NP_FirstOctaveConv(nn.Module):
#     def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
#                     bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
#         super(M_NP_FirstOctaveConv, self).__init__()
#         assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
#         padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
#         self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         self.h2g_pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         self.stride = stride
#         ###low frequency
#         self.h2l = nn.Conv2d(in_nc, int(alpha * out_nc),
#                                 kernel_size, 1, padding, dilation, groups, bias)
#         ###high frequency
#         self.h2h = nn.Conv2d(in_nc, out_nc - int(alpha * out_nc),
#                                 kernel_size, 1, padding, dilation, groups, bias)
#         self.a = act(act_type) if act_type else None
#         self.n_h = norm(norm_type, int(out_nc*(1 - alpha))) if norm_type else None
#         self.n_l = norm(norm_type, int(out_nc*alpha)) if norm_type else None

#     def forward(self, x):
#         #if self.stride ==2:
#             #x = self.h2g_pool(x)

#         X_h = self.h2h(x)
#         #X_l = self.h2l(self.h2g_pool(x))
#         #X_l = self.h2l(x)#without pool
#         X_l = self.h2l(self.h2g_pool2(self.h2g_pool(x)))#double pool

#         if self.n_h and self.n_l:##batch norm
#             X_h = self.n_h(X_h)
#             X_l = self.n_l(X_l)

#         if self.a:#Activation layer
#             X_h = self.a(X_h)
#             X_l = self.a(X_l)

#         return X_h, X_l


# class M_NP_LastOctaveConv(nn.Module):
#     def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
#                     bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
#         super(M_NP_LastOctaveConv, self).__init__()
#         assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
#         padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
#         self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)
#         self.h2g_pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsample = nn.Upsample(scale_factor=4, mode='nearest')##double pool
#         self.stride = stride

#         self.l2h = nn.Conv2d(int(alpha * in_nc), out_nc,
#                                 kernel_size, 1, padding, dilation, groups, bias)
#         self.h2h = nn.Conv2d(in_nc - int(alpha * in_nc), out_nc,
#                                 kernel_size, 1, padding, dilation, groups, bias)

#         self.a = act(act_type) if act_type else None
#         self.n_h = norm(norm_type, out_nc) if norm_type else None

#     def forward(self, x):
#         X_h, X_l = x

#         #if self.stride ==2:
#             #X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)
        
#         X_h2h = self.h2h(X_h)
#         X_l2h = self.upsample(self.l2h(X_l))
#         #X_l2h = self.l2h(X_l)
        
#         X_h = X_h2h + X_l2h

#         if self.n_h:
#             X_h = self.n_h(X_h)

#         if self.a:
#             X_h = self.a(X_h)

#         return X_h

# class M_NP_octave_ResidualDenseBlockTiny_4C(nn.Module):
#     '''
#     Residual Dense Block
#     style: 4 convs
#     The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
#     '''

#     def __init__(self, nc, kernel_size=3, gc=16,alpha=0.5, stride=1, bias=True, pad_type='zero', \
#             norm_type=None, act_type='leakyrelu', mode='CNA'):
#         super(M_NP_octave_ResidualDenseBlockTiny_4C, self).__init__()
#         # gc: growth channel, i.e. intermediate channels
#         self.conv1 =M_NP_OctaveConv(nc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
#              norm_type=norm_type, act_type=act_type, mode=mode) 
#         # conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
#         #     norm_type=norm_type, act_type=act_type, mode=mode)
#         self.conv2 = M_NP_OctaveConv(nc+gc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
#              norm_type=norm_type, act_type=act_type, mode=mode) 
#         # conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
#         #     norm_type=norm_type, act_type=act_type, mode=mode)
#         self.conv3 = M_NP_OctaveConv(nc+2*gc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
#              norm_type=norm_type, act_type=act_type, mode=mode) 
#         # conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
#         #     norm_type=norm_type, act_type=act_type, mode=mode)
#         if mode == 'CNA':
#             last_act = None
#         else:
#             last_act = act_type
#         self.conv4 = M_NP_OctaveConv(nc+3*gc, nc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
#              norm_type=norm_type, act_type=act_type, mode=mode) 
#         # conv_block(nc+3*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
#         #     norm_type=norm_type, act_type=last_act, mode=mode)

#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2((torch.cat((x[0], x1[0]), dim=1),(torch.cat((x[1], x1[1]), dim=1))))
#         x3 = self.conv3((torch.cat((x[0], x1[0],x2[0]), dim=1),(torch.cat((x[1], x1[1],x2[1]), dim=1))))
#         x4 = self.conv4((torch.cat((x[0], x1[0],x2[0],x3[0]), dim=1),(torch.cat((x[1], x1[1],x2[1],x3[1]), dim=1))))

#         res = (x4[0].mul(0.2), x4[1].mul(0.2))
#         x = (x[0] + res[0], x[1] + res[1])
#         #print(len(x),"~~~",len(res),"~~~",len(x + res))

#         #return (x[0] + res[0], x[1]+res[1])
#         return x


# class M_NP_octave_RRDBTiny(nn.Module):
#     '''
#     Residual in Residual Dense Block
#     (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
#     '''

#     def __init__(self, nc, kernel_size=3, gc=16, stride=1, alpha=0.5, bias=True, pad_type='zero', \
#             norm_type=None, act_type='leakyrelu', mode='CNA'):
#         super(M_NP_octave_RRDBTiny, self).__init__()
#         self.RDB1 = M_NP_octave_ResidualDenseBlockTiny_4C(nc=nc, kernel_size=kernel_size,alpha=alpha, gc=gc, stride=stride, bias=bias, pad_type=pad_type, \
#             norm_type=norm_type, act_type=act_type, mode=mode)
#         self.RDB2 = M_NP_octave_ResidualDenseBlockTiny_4C(nc=nc, kernel_size=kernel_size,alpha=alpha, gc=gc, stride=stride, bias=bias, pad_type=pad_type, \
#             norm_type=norm_type, act_type=act_type, mode=mode)

#     def forward(self, x):
#         out = self.RDB1(x)
#         out = self.RDB2(out)

#         res = (out[0].mul(0.2), out[1].mul(0.2))
#         x = (x[0] + res[0], x[1] + res[1])
#         #print(len(x),"~~~",len(res),"~~~",len(x + res))

#         #return (x[0] + res[0], x[1]+res[1])
#         return x

##################################################################################
##################################################################################
##################################################################################
#DWT octave
# Block for OctConv
####################
class DWT_OctaveConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
                    bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
        super(DWT_OctaveConv, self).__init__()
        assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
        padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
        self.xfm = DWTForward(J=1, wave='db3', mode='zero').cuda()
        self.ifm = DWTInverse(wave='db3', mode='zero').cuda()
        self.stride = stride
 
        self.l2l = nn.Conv2d(int(alpha * in_nc), int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = nn.Conv2d(int(alpha * in_nc), out_nc - int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = nn.Conv2d(in_nc - int(alpha * in_nc), int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_nc - int(alpha * in_nc), out_nc - int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
 
        self.a = act(act_type) if act_type else None
        self.n_h = norm(norm_type, int(out_nc*(1 - alpha))) if norm_type else None
        self.n_l = norm(norm_type, int(out_nc*alpha)) if norm_type else None
 
 
    def forward(self, x):
        X_h,X_l_ll,X_l_lh,X_l_hl,X_l_hh = x
 
        #for X_h to h
        X_h2h = self.h2h(X_h)
        #X_l2h = self.upsample(self.l2h(X_l))
        #get X_l
        H=X_l_ll
        j= H.shape
        J = torch.randn(j[0], j[1], 3, j[-2], j[-1])
        J[:,:,0]=X_l_lh
        J[:,:,1]=X_l_hl
        J[:,:,2]=X_l_hh
        J_ = [J.cuda()]
        X_l=self.ifm((H,J_))
        #X_l to h
        X_l2h = self.l2h(X_l)
 
 
        #DWT for X_h
        A,B=self.xfm(X_h)
        X_hll=A
        X_hlh=B[0][:,:,0]
        X_hhl=B[0][:,:,1]
        X_hhh=B[0][:,:,2]
 
        #transfer
        X_hll2l=self.h2l(X_hll)
        X_hlh2l=self.h2l(X_hlh)
        X_hhl2l=self.h2l(X_hhl)
        X_hhh2l=self.h2l(X_hhh)
 
        #for X_l series (X_l_ll,X_l_lh,X_l_hl,X_l_hh)
        #transfer
 
        X_lll2l=self.l2l(X_l_ll)
        X_llh2l=self.l2l(X_l_lh)
        X_lhl2l=self.l2l(X_l_hl)
        X_lhh2l=self.l2l(X_l_hh)
 
 
        #for X_h
        X_h=X_h2h+X_l2h
 
        #for X_l series (X_l_ll,X_l_lh,X_l_hl,X_l_hh)
 
        X_l_ll=X_lll2l+X_hll2l
        X_l_lh=X_llh2l+X_hlh2l
        X_l_hl=X_lhl2l+X_hhl2l
        X_l_hh=X_lhh2l+X_hhh2l
 
 
 
        if self.n_h and self.n_l:
            X_h = self.n_h(X_h)
            #X_l = self.n_l(X_l)
            X_l_ll = self.n_l(X_l_ll)
            X_l_lh = self.n_l(X_l_lh)
            X_l_hl = self.n_l(X_l_hl)
            X_l_hh = self.n_l(X_l_hh)
 
        if self.a:
            X_h = self.a(X_h)
            #X_l = self.a(X_l)
            X_l_ll = self.a(X_l_ll)
            X_l_lh = self.a(X_l_lh)
            X_l_hl = self.a(X_l_hl)
            X_l_hh = self.a(X_l_hh)
 
 
        return X_h,X_l_ll,X_l_lh,X_l_hl,X_l_hh
 
 
 
class DWT_FirstOctaveConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
                    bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
        super(DWT_FirstOctaveConv, self).__init__()
        assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
        padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
        self.xfm = DWTForward(J=1, wave='db3', mode='zero').cuda()
        #self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        #self.h2g_pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        #self.xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b').cuda()
        #self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b').cuda()
        self.stride = stride
        ###low frequency
        self.h2l = nn.Conv2d(in_nc, int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        ###high frequency
        self.h2h = nn.Conv2d(in_nc, out_nc - int(alpha * out_nc),
                                kernel_size, 1, padding, dilation, groups, bias)
        self.a = act(act_type) if act_type else None
        self.n_h = norm(norm_type, int(out_nc*(1 - alpha))) if norm_type else None
        self.n_l = norm(norm_type, int(out_nc*alpha)) if norm_type else None
 
    def forward(self, x):
        #if self.stride ==2:
            #x = self.h2g_pool(x)
 
        X_h = self.h2h(x)
        X_l = self.h2l(x)
 
        A,B=self.xfm(X_l)
        X_l_ll=A
        X_l_lh=B[0][:,:,0]
        X_l_hl=B[0][:,:,1]
        X_l_hh=B[0][:,:,2]
 
 
        if self.n_h and self.n_l:##batch norm
            X_h = self.n_h(X_h)
            #X_l = self.n_l(X_l)
            X_l_ll = self.n_l(X_l_ll)
            X_l_lh = self.n_l(X_l_lh)
            X_l_hl = self.n_l(X_l_hl)
            X_l_hh = self.n_l(X_l_hh)
 
 
        if self.a:#Activation layer
            X_h = self.a(X_h)
            #X_l = self.a(X_l)
            X_l_ll = self.a(X_l_ll)
            X_l_lh = self.a(X_l_lh)
            X_l_hl = self.a(X_l_hl)
            X_l_hh = self.a(X_l_hh)
 
        return X_h,X_l_ll,X_l_lh,X_l_hl,X_l_hh
 
 
class DWT_LastOctaveConv(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size, alpha=0.5, stride=1, dilation=1, groups=1, \
                    bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA'):
        super(DWT_LastOctaveConv, self).__init__()
        assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
        padding = get_valid_padding(kernel_size, dilation) if pad_type == 'zero' else 0
        #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        #self.upsample = nn.Upsample(scale_factor=4, mode='nearest')##double pool
        self.ifm = DWTInverse(wave='db3', mode='zero').cuda()
        self.stride = stride
 
        self.l2h = nn.Conv2d(int(alpha * in_nc), out_nc,
                                kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = nn.Conv2d(in_nc - int(alpha * in_nc), out_nc,
                                kernel_size, 1, padding, dilation, groups, bias)
 
        self.a = act(act_type) if act_type else None
        self.n_h = norm(norm_type, out_nc) if norm_type else None
 
    def forward(self, x):
        X_h,X_l_ll,X_l_lh,X_l_hl,X_l_hh = x
 
        #if self.stride ==2:
            #X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)
 
        A=X_l_ll
        c= A.shape
        C = torch.randn(c[0], c[1], 3, c[-2], c[-1])
        #X_lh_ = torch.unsqueeze(X_lh, 2)
        #X_hl_ = torch.unsqueeze(X_hl, 2)
        #X_hh_ = torch.unsqueeze(X_hh, 2)
        #C=torch.cat((X_lh_,X_hl_,X_hh_), dim=2)
        C[:,:,0]=X_l_lh
        C[:,:,1]=X_l_hl
        C[:,:,2]=X_l_hh
        #C=C.cpu()
        C_ = [C.cuda()]
        #A=A.cpu()
        X_l=self.ifm((A,C_))
 
 
        X_h2h = self.h2h(X_h)
        X_l2h=self.l2h(X_l)
        #X_l2h = self.l2h(X_l)
        
        X_h = X_h2h + X_l2h
 
        if self.n_h:
            X_h = self.n_h(X_h)
 
        if self.a:
            X_h = self.a(X_h)
 
        return X_h
 
class DWT_octave_ResidualDenseBlockTiny_4C(nn.Module):
    '''
    Residual Dense Block
    style: 4 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''
 
    def __init__(self, nc, kernel_size=3, gc=16,alpha=0.5, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(DWT_octave_ResidualDenseBlockTiny_4C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 =DWT_OctaveConv(nc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
             norm_type=norm_type, act_type=act_type, mode=mode) 
        # conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = DWT_OctaveConv(nc+gc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
             norm_type=norm_type, act_type=act_type, mode=mode) 
        # conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = DWT_OctaveConv(nc+2*gc, gc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
             norm_type=norm_type, act_type=act_type, mode=mode) 
        # conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv4 = DWT_OctaveConv(nc+3*gc, nc, kernel_size, alpha, stride, bias=bias, pad_type=pad_type, \
             norm_type=norm_type, act_type=act_type, mode=mode) 
        # conv_block(nc+3*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
        #     norm_type=norm_type, act_type=last_act, mode=mode)
 
    def forward(self, x):
        # print(x[0].shape,'~~~',x[1].shape,'~~~',x[2].shape,'~~~',x[3].shape)
        # exit()
        x1 = self.conv1(x)
        x2 = self.conv2((torch.cat((x[0], x1[0]), dim=1),torch.cat((x[1], x1[1]), dim=1),torch.cat((x[2], x1[2]), dim=1),torch.cat((x[3], x1[3]), dim=1),torch.cat((x[4], x1[4]), dim=1)))
        x3 = self.conv3((torch.cat((x[0], x1[0],x2[0]), dim=1),torch.cat((x[1], x1[1],x2[1]), dim=1),torch.cat((x[2], x1[2],x2[2]), dim=1),torch.cat((x[3], x1[3],x2[3]), dim=1),torch.cat((x[4], x1[4],x2[4]), dim=1)))
        x4 = self.conv4((torch.cat((x[0], x1[0],x2[0],x3[0]), dim=1),torch.cat((x[1], x1[1],x2[1],x3[1]), dim=1),torch.cat((x[2], x1[2],x2[2],x3[2]), dim=1),torch.cat((x[3], x1[3],x2[3],x3[3]), dim=1),torch.cat((x[4], x1[4],x2[4],x3[4]), dim=1)))
 
 
        res = (x4[0].mul(0.2), x4[1].mul(0.2),x4[2].mul(0.2),x4[3].mul(0.2),x4[4].mul(0.2))
        x = (x[0] + res[0], x[1] + res[1], x[2] + res[2], x[3] + res[3], x[4] + res[4])
        #print(len(x),"~~~",len(res),"~~~",len(x + res))
 
        #return (x[0] + res[0], x[1]+res[1])
        return x
 
 
 
class DWT_octave_RRDBTiny(nn.Module):
    '''
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    '''
 
    def __init__(self, nc, kernel_size=3, gc=16, stride=1, alpha=0.5, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(DWT_octave_RRDBTiny, self).__init__()
        self.RDB1 = DWT_octave_ResidualDenseBlockTiny_4C(nc=nc, kernel_size=kernel_size,alpha=alpha, gc=gc, stride=stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
        self.RDB2 = DWT_octave_ResidualDenseBlockTiny_4C(nc=nc, kernel_size=kernel_size,alpha=alpha, gc=gc, stride=stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, act_type=act_type, mode=mode)
 
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
 
        res = (out[0].mul(0.2), out[1].mul(0.2),out[2].mul(0.2),out[3].mul(0.2),out[4].mul(0.2))
        x = (x[0] + res[0], x[1] + res[1], x[2] + res[2], x[3] + res[3],x[4] + res[4])
 
        #print(len(x),"~~~",len(res),"~~~",len(x + res))
 
        #return (x[0] + res[0], x[1]+res[1])
        return x