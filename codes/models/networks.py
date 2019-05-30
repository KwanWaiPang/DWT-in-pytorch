import functools
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch
import models.modules.sft_arch as sft_arch

####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)
    # elif classname.find('AdaptiveConvResNorm') != -1:
    #     init.constant_(m.weight.data, 0.0)
    #     if m.bias is not None:
    #         m.bias.data.zero_()


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################

# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'sr_resnet':  # SRResNet
        netG = arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'Octave_SRResNet':  # SRResNet based on octave
        netG = arch.Octave_SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'M_NP_Octave_RRDBNet':  # SRResNet based on octave
        netG = arch.M_NP_Octave_RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'Octave_RRDBNet':  # SRResNet based on octave
        netG = arch.Octave_RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'DWT_Octave_RRDBNet':  # SRResNet based on octave
        netG = arch.DWT_Octave_RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'modified_resnet':  # SRResNet based on octave
        netG = arch.Modified_SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'Octave_CARN':  # CARN based on octave
        netG = arch.Octave_CARN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'carn':  # CARN based on octave
        netG = arch.CARN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle')

    elif which_model == 'modulate_sr_resnet':
        netG = arch.ModulateSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                     upscale=opt_net['scale'], norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                     upsample_mode='pixelshuffle', ada_ksize=opt_net['ada_ksize'],
                                     gate_conv_bias=opt_net['gate_conv_bias'])

    elif which_model == 'arcnn':
        netG = arch.ARCNN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                             norm_type=opt_net['norm_type'], mode=opt_net['mode'], ada_ksize=opt_net['ada_ksize'])

    elif which_model == 'srcnn':
        netG = arch.SRCNN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                             norm_type=opt_net['norm_type'], mode=opt_net['mode'], ada_ksize=opt_net['ada_ksize'])

    elif which_model == 'noise_plainnet':
        netG = arch.NoisePlainNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
                                  norm_type=opt_net['norm_type'], mode=opt_net['mode'])

    elif which_model == 'denoise_resnet':
        netG = arch.DenoiseResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                  upscale=opt_net['scale'], norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                  upsample_mode='pixelshuffle', ada_ksize=opt_net['ada_ksize'],
                                  down_scale=opt_net['down_scale'], fea_norm=opt_net['fea_norm'],
                                  upsample_norm=opt_net['upsample_norm'])
    elif which_model == 'modulate_denoise_resnet':
        netG = arch.ModulateDenoiseResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                          upscale=opt_net['scale'], norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                          upsample_mode='pixelshuffle', ada_ksize=opt_net['ada_ksize'],
                                          gate_conv_bias=opt_net['gate_conv_bias'])
    elif which_model == 'noise_subnet':
        netG = arch.NoiseSubNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                norm_type=opt_net['norm_type'], mode=opt_net['mode'])
    elif which_model == 'cond_denoise_resnet':
        netG = arch.CondDenoiseResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                      upscale=opt_net['scale'], upsample_mode='pixelshuffle', ada_ksize=opt_net['ada_ksize'],
                                      down_scale=opt_net['down_scale'], num_classes=opt_net['num_classes'],
                                      norm_type=opt_net['norm_type'])

    elif which_model == 'adabn_denoise_resnet':
        netG = arch.AdaptiveDenoiseResNet(in_nc=opt_net['in_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                          upscale=opt_net['scale'], down_scale=opt_net['down_scale'])

    elif which_model == 'sft_arch':  # SFT-GAN
        netG = sft_arch.SFT_Net()

    elif which_model == 'RRDB_net':  # RRDB
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv')
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['init_type'] is not None:
        init_weights(netG, init_type=opt['init_type'], scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

##################################################################################################################
#define subnetwork
def define_sub(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_sub']
    which_model = opt_net['which_model_sub']

    if which_model == 'noise_subnet':
        subnet = arch.NoiseSubNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                norm_type=opt_net['norm_type'], mode=opt_net['mode'])
    else:
        raise NotImplementedError('subnet model [{:s}] not recognized'.format(which_model))

    if gpu_ids:
        assert torch.cuda.is_available()
        subnet = nn.DataParallel(subnet)
    return subnet

def define_sub2(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_sub2']
    which_model = opt_net['which_model_sub']

    if which_model == 'blur_subnet':
        subnet = arch.NoiseSubNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                norm_type=opt_net['norm_type'], mode=opt_net['mode'])
    elif which_model == 'denoise_resnet':
        subnet = arch.DenoiseResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'],
                                    upscale=opt_net['scale'], norm_type=opt_net['norm_type'], mode=opt_net['mode'],
                                    upsample_mode='pixelshuffle', ada_ksize=opt_net['ada_ksize'],
                                    down_scale=opt_net['down_scale'], fea_norm=opt_net['fea_norm'],
                                    upsample_norm=opt_net['upsample_norm'])
    else:
        raise NotImplementedError('subnet model [{:s}] not recognized'.format(which_model))

    if gpu_ids:
        assert torch.cuda.is_available()
        subnet = nn.DataParallel(subnet)
    return subnet

# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])

    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = sft_arch.ACD_VGG_BN_96()

    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD


def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF
