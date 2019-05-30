import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from . import block as B
from . import spectral_norm as SN
from . import adaptive_norm as AN

####################
# Generator
##############################################################################################
#octave_srresnet
class Octave_SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4,alpha=0.75, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(Octave_SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type='relu', mode='CNA')
        
        fea_conv1 = B.M_NP_FirstOctaveConv(nf, nf, kernel_size=3, alpha=alpha, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA')
        #fea_conv1 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='relu', mode='CNA')
        
        resnet_blocks = [B.M_NP_OctaveResBlock(nf, nf, nf, kernel_size=3,alpha=alpha,norm_type=norm_type, act_type=act_type, mode=mode, res_scale=res_scale) for _ in range(nb)]
        #resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type, mode=mode, res_scale=res_scale) for _ in range(nb)]
        
        LR_conv = B.M_NP_LastOctaveConv(nf, nf, kernel_size=3, alpha=alpha, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA')
        #LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='relu', mode='CNA')

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv,B.ShortcutBlock(B.sequential(fea_conv1,*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
##################this is ESRGAN based on Moctave
class M_NP_Octave_RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32,alpha=0.75, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(M_NP_Octave_RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv1 = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        fea_conv = B.M_NP_FirstOctaveConv(nf, nf, kernel_size=3,alpha=alpha, stride=1, dilation=1, groups=1, \
                    bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA')
        rb_blocks = [B.M_NP_octave_RRDBTiny(nf, kernel_size=3, gc=32,alpha=alpha,stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.M_NP_LastOctaveConv(nf, nf, kernel_size=3, alpha=alpha, stride=1, dilation=1, groups=1, \
                    bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA')

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv1,B.ShortcutBlock(B.sequential(fea_conv,*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
##############################################################################################


##################this is ESRGAN based on octave
class Octave_RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32,alpha=0.75, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(Octave_RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv1 = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # fea_conv = B.FirstOctaveConv(nf, nf, kernel_size=3,alpha=alpha, stride=1, dilation=1, groups=1, \
        #             bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA')
        fea_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)
        # rb_blocks = [B.octave_RRDBTiny(nf, kernel_size=3, gc=32,alpha=alpha,stride=1, bias=True, pad_type='zero', \
        #     norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        rb_blocks = [B.RRDBTiny(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        # LR_conv = B.LastOctaveConv(nf, nf, kernel_size=3, alpha=alpha, stride=1, dilation=1, groups=1, \
        #             bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA')
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv1,B.ShortcutBlock(B.sequential(fea_conv,*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
##############################################################################################

##################this is ESRGAN
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
        #     norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        rb_blocks = [B.RRDBTiny(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x
##############################################################################################

##############################################################################################
#CARN
class CARN(nn.Module):#nb=3(3 block),channel=24
    def __init__(self, in_nc, out_nc, nf=24, nc=4, nb=3, alpha=0.75, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(CARN, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.nb = nb

        self.fea_conv =B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        self.CascadeBlocks = nn.ModuleList([B.CascadeBlock(nf, nf, kernel_size=3, norm_type=norm_type, \
            act_type=act_type, mode=mode, res_scale=res_scale) for _ in range(nb)])
        self.CatBlocks = nn.ModuleList([B.conv_block((i + 2)*nf, nf, kernel_size=1, \
            norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nb)])
        self.P_conv = B.conv_block(nf, in_nc*(upscale ** 2), kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        self.upsampler = nn.PixelShuffle(upscale)



    def forward(self, x):
        x = self.fea_conv(x)
        pre_fea = x
        for i in range(self.nb):
            res = self.CascadeBlocks[i](x)
            pre_fea = torch.cat((pre_fea, res), dim=1)
            x = self.CatBlocks[i](pre_fea)
        x = self.P_conv(x)
        x = F.sigmoid(self.upsampler(x))
        return x


##############################################################################################
##############################################################################################
#Octave CARN
class Octave_CARN(nn.Module):#nb=3(3 block),channel=24
    def __init__(self, in_nc, out_nc, nf=24, nc=4, nb=3, alpha=0.75, upscale=4, norm_type=None, act_type='prelu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(Octave_CARN, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.nb = nb

        self.fea_conv =B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode='CNA')

        ##################################################################
        self.oct_first=B.M_NP_FirstOctaveConv(nf, nf, kernel_size=3,  alpha=alpha, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA')
        #self.oct_first =B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='prelu', mode='CNA')
        ##################################################################
        self.CascadeBlocks = nn.ModuleList([B.M_NP_OctaveCascadeBlock(nc, nf, kernel_size=3, alpha=alpha, norm_type=norm_type, act_type=act_type, mode=mode, res_scale=res_scale) for _ in range(nb)])
        #self.CascadeBlocks = nn.ModuleList([B.CascadeBlock(nc, nf, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode, res_scale=res_scale) for _ in range(nb)])
        ##################################################################
        self.CatBlocks = nn.ModuleList([B.M_NP_OctaveConv((i + 2)*nf, nf, kernel_size=1, alpha=alpha, norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nb)])
        #self.CatBlocks = nn.ModuleList([B.conv_block((i + 2)*nf, nf, kernel_size=1, norm_type=norm_type, act_type=act_type, mode=mode) for i in range(nb)])
        ##################################################################
        self.oct_last = B.M_NP_LastOctaveConv(nf, nf, kernel_size=3, alpha=alpha, stride=1, dilation=1, groups=1, bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA')
        #self.oct_last = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type='prelu', mode='CNA')
        ##################################################################
        self.upsampler = nn.PixelShuffle(upscale)
        self.HR_conv1 = B.conv_block(nf, in_nc*(upscale ** 2), kernel_size=3, norm_type=None, act_type=None)


    def forward(self, x):
        x = self.fea_conv(x)
        x = self.oct_first(x)
        pre_fea = x
        for i in range(self.nb):
            res = self.CascadeBlocks[i](x)
            pre_fea = (torch.cat((pre_fea[0], res[0]), dim=1), \
                        torch.cat((pre_fea[1], res[1]), dim=1))
            x = self.CatBlocks[i](pre_fea)
        # for i in range(self.nb):
        #     res = self.CascadeBlocks[i](x)
        #     pre_fea = torch.cat((pre_fea, res), dim=1)
        #     x = self.CatBlocks[i](pre_fea)
        x = self.oct_last(x)
        x = self.HR_conv1(x)
        x = F.sigmoid(self.upsampler(x))
        return x


# ##############################################################################################
# #Ocatve SRResNet
# class Octave_SRResNet(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
#             mode='NAC', res_scale=1, upsample_mode='upconv'):
#         super(Octave_SRResNet, self).__init__()
#         n_upscale = int(math.log(upscale, 2))
#         if upscale == 3:
#             n_upscale = 1

#         fea_conv = B.FirstOctaveConv(in_nc, nf, kernel_size=3,  alpha=0.75, stride=1, dilation=1, groups=1, \
#                     bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA')
#         #fea_conv=B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
#         resnet_blocks = [B.OctaveResBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
#             mode=mode, res_scale=res_scale) for _ in range(nb)]
#         #LR_conv=B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
#         LR_conv = B.LastOctaveConv(nf, nf, kernel_size=3, alpha=0.75, stride=1, dilation=1, groups=1, \
#                     bias=True, pad_type='zero', norm_type=None, act_type='prelu', mode='CNA')

#         if upsample_mode == 'upconv':
#             upsample_block = B.upconv_blcok
#         elif upsample_mode == 'pixelshuffle':
#             upsample_block = B.pixelshuffle_block
#         else:
#             raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
#         if upscale == 3:
#             upsampler = upsample_block(nf, nf, 3, act_type=act_type)
#         else:
#             upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
#         HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
#         HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

#         self.model = B.sequential(fea_conv, *resnet_blocks, LR_conv,\
#             *upsampler, HR_conv0, HR_conv1)

#     def forward(self, x):
#         x = self.model(x)
#         return x

##############################################################################################
#Modified SRResNet
class Modified_SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(Modified_SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv=B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv=B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, *resnet_blocks, LR_conv,\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


#####################################################################################################

#####################################################################################################
class SRCNN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, norm_type='batch', act_type='relu', mode='CNA', ada_ksize=None):
        super(SRCNN, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=9, norm_type=norm_type, act_type=act_type, mode=mode
                                , ada_ksize=ada_ksize)
        mapping_conv = B.conv_block(nf, nf // 2, kernel_size=1, norm_type=norm_type, act_type=act_type,
                                    mode=mode, ada_ksize=ada_ksize)
        HR_conv = B.conv_block(nf // 2, out_nc, kernel_size=5, norm_type=norm_type, act_type=None,
                               mode=mode, ada_ksize=ada_ksize)

        self.model = B.sequential(fea_conv, mapping_conv, HR_conv)

    def forward(self, x):
        x = self.model(x)
        return x


class ARCNN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, norm_type='batch', act_type='relu', mode='CNA', ada_ksize=None):
        super(ARCNN, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=9, norm_type=norm_type, act_type=act_type, mode=mode
                                , ada_ksize=ada_ksize)
        conv1 = B.conv_block(nf, nf // 2, kernel_size=7, norm_type=norm_type, act_type=act_type,
                             mode=mode, ada_ksize=ada_ksize)
        conv2 = B.conv_block(nf // 2, nf // 4, kernel_size=1, norm_type=norm_type, act_type=act_type,
                             mode=mode, ada_ksize=ada_ksize)
        HR_conv = B.conv_block(nf // 4, out_nc, kernel_size=5, norm_type=norm_type, act_type=None,
                               mode=mode, ada_ksize=ada_ksize)

        self.model = B.sequential(fea_conv, conv1, conv2, HR_conv)

    def forward(self, x):
        x = self.model(x)
        return x


class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class ModulateSRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='sft', act_type='relu',
                 mode='CNA', res_scale=1, upsample_mode='upconv', gate_conv_bias=True, ada_ksize=None):
        super(ModulateSRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, stride=1)
        resnet_blocks = [B.TwoStreamSRResNet(nf, nf, nf, norm_type=norm_type, act_type=act_type,
                         mode=mode, res_scale=res_scale, gate_conv_bias=gate_conv_bias,
                                             ada_ksize=ada_ksize, input_dim=in_nc) for _ in range(nb)]

        self.LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode)
        if norm_type == 'sft':
            self.LR_norm = AN.GateNonLinearLayer(in_nc, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            self.LR_norm = AN.MetaLayer(in_nc, conv_bias=gate_conv_bias, kernel_size=ada_ksize)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.norm_branch = B.sequential(*resnet_blocks)
        self.HR_branch = B.sequential(*upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        fea = self.fea_conv(x[0])
        fea_res_block, _ = self.norm_branch((fea, x[1]))
        fea_LR = self.LR_conv(fea_res_block)
        res = self.LR_norm((fea_LR, x[1]))
        out = self.HR_branch(fea+res)
        return out


class DenoiseResNet(nn.Module):
    """
    jingwen's addition
    denoise Resnet
    """
    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, norm_type='batch', act_type='relu',
                 mode='CNA', res_scale=1, upsample_mode='upconv', ada_ksize=None, down_scale=2,
                 fea_norm=None, upsample_norm=None):
        super(DenoiseResNet, self).__init__()
        n_upscale = int(math.log(down_scale, 2))
        if down_scale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=fea_norm, act_type=None, stride=down_scale,
                                ada_ksize=ada_ksize)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,
                         mode=mode, res_scale=res_scale, ada_ksize=ada_ksize) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode
                               , ada_ksize=ada_ksize)
        # LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode
        #                        , ada_ksize=ada_ksize)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)

        if down_scale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type, norm_type=upsample_norm, ada_ksize=ada_ksize)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type, norm_type=upsample_norm, ada_ksize=ada_ksize) for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=upsample_norm, act_type=act_type, ada_ksize=ada_ksize)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=upsample_norm, act_type=None, ada_ksize=ada_ksize)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),
                                  *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class ModulateDenoiseResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, norm_type='sft', act_type='relu',
                 mode='CNA', res_scale=1, upsample_mode='upconv', gate_conv_bias=True, ada_ksize=None):
        super(ModulateDenoiseResNet, self).__init__()

        self.fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, stride=2)
        resnet_blocks = [B.TwoStreamSRResNet(nf, nf, nf, norm_type=norm_type, act_type=act_type,
                         mode=mode, res_scale=res_scale, gate_conv_bias=gate_conv_bias,
                                             ada_ksize=ada_ksize, input_dim=in_nc) for _ in range(nb)]

        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode)
        if norm_type == 'sft':
            LR_norm = AN.GateNonLinearLayer(in_nc, conv_bias=gate_conv_bias)
        elif norm_type == 'sft_conv':
            LR_norm = AN.MetaLayer(in_nc, conv_bias=gate_conv_bias, kernel_size=ada_ksize)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        upsampler = upsample_block(nf, nf, act_type=act_type)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.norm_branch = B.sequential(*resnet_blocks)
        self.LR_conv = LR_conv
        self.LR_norm = LR_norm
        self.HR_branch = B.sequential(upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        fea = self.fea_conv(x[0])
        fea_res_block, _ = self.norm_branch((fea, x[1]))
        fea_LR = self.LR_conv(fea_res_block)
        res = self.LR_norm((fea_LR, x[1]))
        out = self.HR_branch(fea+res)
        return out


class NoiseSubNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, norm_type='batch', act_type='relu', mode='CNA'):
        super(NoiseSubNet, self).__init__()
        degration_block = [B.conv_block(in_nc, nf, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode)]
        degration_block.extend([B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode)
                                for _ in range(15)])
        degration_block.append(B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None, mode=mode))
        self.degration_block = B.sequential(*degration_block)

    def forward(self, x):
        deg_estimate = self.degration_block(x)
        return deg_estimate

class NoisePlainNet(nn.Module):
    """
    jingwen's addition
    denoise Resnet
    """
    def __init__(self, in_nc, out_nc, nf, upscale=1, norm_type='batch', act_type='leakyrelu',
                 mode='CNA'):
        super(NoisePlainNet, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=act_type)
        middle_blocks = [B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type,
                         mode=mode) for _ in range(3)]
        sigma_conv = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.sequential(*middle_blocks), sigma_conv)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class CondDenoiseResNet(nn.Module):
    """
    jingwen's addition
    denoise Resnet
    """

    def __init__(self, in_nc, out_nc, nf, nb, upscale=1, res_scale=1, down_scale=2, num_classes=1, ada_ksize=None
                 ,upsample_mode='upconv', act_type='relu', norm_type='cond_adaptive_conv_res'):
        super(CondDenoiseResNet, self).__init__()
        n_upscale = int(math.log(down_scale, 2))
        if down_scale == 3:
            n_upscale = 1

        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=down_scale, padding=1)
        resnet_blocks = [B.CondResNetBlock(nf, nf, nf, num_classes=num_classes, ada_ksize=ada_ksize,
                                           norm_type=norm_type, act_type=act_type) for _ in range(nb)]
        self.resnet_blocks = B.sequential(*resnet_blocks)
        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        if norm_type == 'cond_adaptive_conv_res':
            self.cond_adaptive = AN.CondAdaptiveConvResNorm(nf, num_classes=num_classes)
        elif norm_type == "interp_adaptive_conv_res":
            self.cond_adaptive = AN.InterpAdaptiveResNorm(nf, ada_ksize)
        elif norm_type == "cond_instance":
            self.cond_adaptive = AN.CondInstanceNorm2d(nf, num_classes=num_classes)
        elif norm_type == "cond_transform_res":
            self.cond_adaptive = AN.CondResTransformer(nf, ada_ksize, num_classes=num_classes)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)

        if down_scale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.upsample = B.sequential(*upsampler, HR_conv0, HR_conv1)

    def forward(self, x, y):
        # the first feature extraction
        fea = self.fea_conv(x)
        fea1, _ = self.resnet_blocks((fea, y))
        fea2 = self.LR_conv(fea1)
        fea3 = self.cond_adaptive(fea2, y)
        # res
        out = self.upsample(fea3 + fea)
        return out


class AdaptiveDenoiseResNet(nn.Module):
    """
    jingwen's addition
    adabn
    """
    def __init__(self, in_nc, nf, nb, upscale=1, res_scale=1, down_scale=2):
        super(AdaptiveDenoiseResNet, self).__init__()

        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=down_scale, padding=1)
        resnet_blocks = [B.AdaptiveResNetBlock(nf, nf, nf, res_scale=res_scale) for _ in range(nb)]
        self.resnet_blocks = B.sequential(*resnet_blocks)
        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(nf, affine=True, track_running_stats=True, momentum=0)

    def forward(self, x):
        fea_list = [self.fea_conv(data.unsqueeze_(0)) for data in x]
        fea_resblock_list = self.resnet_blocks(fea_list)
        fea_LR_list = [self.LR_conv(fea) for fea in fea_resblock_list]
        fea_mean, fea_var = B.computing_mean_variance(fea_LR_list)

        batch_norm_dict = self.batch_norm.state_dict()
        batch_norm_dict['running_mean'] = fea_mean
        batch_norm_dict['running_var'] = fea_var
        self.batch_norm.load_state_dict(batch_norm_dict)
        return None


