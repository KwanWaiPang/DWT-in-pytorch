import torch
from torch.nn import init
from collections import OrderedDict

pretrained_net = torch.load('../../baselines_jpeg/experiments/JPEG30_gray_nonorm_denoise_resnet_DIV2K/models/982000_G.pth')
# should run train debug mode first to get an initial model
adaptive_norm_net = OrderedDict()


adaptive_norm_net['model.0.weight'] = pretrained_net['model.0.weight']
adaptive_norm_net['model.0.bias'] = pretrained_net['model.0.bias']
# residual blocks
for i in range(16):
    adaptive_norm_net['model.1.sub.{:d}.res.0.weight'.format(i)] = pretrained_net['model.1.sub.{:d}.res.0.weight'.format(i)]
    adaptive_norm_net['model.1.sub.{:d}.res.0.bias'.format(i)] = pretrained_net['model.1.sub.{:d}.res.0.bias'.format(i)]
    adaptive_norm_net['model.1.sub.{:d}.res.3.weight'.format(i)] = pretrained_net['model.1.sub.{:d}.res.2.weight'.format(i)]
    adaptive_norm_net['model.1.sub.{:d}.res.3.bias'.format(i)] = pretrained_net['model.1.sub.{:d}.res.2.bias'.format(i)]

adaptive_norm_net['model.1.sub.16.weight'] = pretrained_net['model.1.sub.16.weight']
adaptive_norm_net['model.1.sub.16.bias'] = pretrained_net['model.1.sub.16.bias']

# HR
adaptive_norm_net['model.2.weight'] = pretrained_net['model.2.weight']
adaptive_norm_net['model.2.bias'] = pretrained_net['model.2.bias']
adaptive_norm_net['model.5.weight'] = pretrained_net['model.5.weight']
adaptive_norm_net['model.5.bias'] = pretrained_net['model.5.bias']
adaptive_norm_net['model.7.weight'] = pretrained_net['model.7.weight']
adaptive_norm_net['model.7.bias'] = pretrained_net['model.7.bias']

print('OK. \n Saving model...')
torch.save(adaptive_norm_net, '../../experiments/pretrained_models/jpeg10to40_models/JPEG30_gray_nonorm_denoise_resnet_DIV2K/jpeg_CNA_adaptive_982000.pth')
