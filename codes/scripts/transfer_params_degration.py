import torch
from torch.nn import init
from collections import OrderedDict

# pretrained_net = torch.load('../../baselines/experiments/bicx3_nonorm_denoise_resnet_DIV2K/models/794000_G.pth')
# pretrained_net = torch.load('../../baselines_jpeg/experiments/JPEG80_gray_nonorm_denoise_resnet_DIV2K/models/964000_G.pth')
# pretrained_net = torch.load('../../noise_from15to75/experiments/gaussian_from15to75_resnet_denoise_DIV2K/models/986000_G.pth')
# pretrained_net = torch.load('/home/jwhe/workspace/BasicSR_v3/experiments/pretrained_models/noise_c16s06/bicx4_nonorm_denoise_resnet_DIV2K/992000_G.pth')
# pretrained_net = torch.load('/home/jwhe/workspace/BasicSR_v3/sr_c16s06/experiments/LR_srx4_c16s06_resnet_denoise_DIV2K/models/704000_G.pth')
pretrained_net = torch.load('/home/jwhe/workspace/BasicSR_v3/sr/experiments/LR_srx4_resnet_denoise_DIV2K/models/516000_G.pth')
# should run train debug mode first to get an initial model
# pretrained_net_degradation = torch.load('../../noise_subnet/experiments/noise75_subnet/models/84000_G.pth')
# pretrained_net_degradation = torch.load('../../noise_subnet/experiments/noise15_subnet/models/34000_G.pth')

# reference_net = torch.load('../../noise_estimate/experiments/finetune_75_sft_64_nores_noise_estimate_15_denoise_resnet_DIV2K/models/20000_G.pth')
adaptive_norm_net = OrderedDict()
# initialize the norm with value 0
# for k, v in adaptive_norm_net.items():
#     if 'gamma' in k:
#         print(k, 'gamma')
#         v.fill_(0)
#     elif 'beta' in k:
#         print(k, 'beta')
#         v.fill_(0)

# for k, v in adaptive_norm_net.items():
#     if 'gamma' in k:
#         print(k, 'gamma')
#         v.fill_(0)
#     elif 'beta' in k:
#         print(k, 'beta')
#         v.fill_(0)

adaptive_norm_net['fea_conv.0.weight'] = pretrained_net['model.0.weight']
adaptive_norm_net['fea_conv.0.bias'] = pretrained_net['model.0.bias']
# residual blocks
for i in range(16):
    adaptive_norm_net['norm_branch.{:d}.conv_block0.0.weight'.format(i)] = pretrained_net['model.1.sub.{:d}.res.0.weight'.format(i)]
    adaptive_norm_net['norm_branch.{:d}.conv_block0.0.bias'.format(i)] = pretrained_net['model.1.sub.{:d}.res.0.bias'.format(i)]
    adaptive_norm_net['norm_branch.{:d}.conv_block1.0.weight'.format(i)] = pretrained_net['model.1.sub.{:d}.res.2.weight'.format(i)]
    adaptive_norm_net['norm_branch.{:d}.conv_block1.0.bias'.format(i)] = pretrained_net['model.1.sub.{:d}.res.2.bias'.format(i)]

adaptive_norm_net['LR_conv.0.weight'] = pretrained_net['model.1.sub.16.weight']
adaptive_norm_net['LR_conv.0.bias'] = pretrained_net['model.1.sub.16.bias']

# HR
# adaptive_norm_net['HR_branch.0.weight'] = pretrained_net['model.2.weight']
# adaptive_norm_net['HR_branch.0.bias'] = pretrained_net['model.2.bias']
# adaptive_norm_net['HR_branch.3.weight'] = pretrained_net['model.5.weight']
# adaptive_norm_net['HR_branch.3.bias'] = pretrained_net['model.5.bias']
# adaptive_norm_net['HR_branch.5.weight'] = pretrained_net['model.7.weight']
# adaptive_norm_net['HR_branch.5.bias'] = pretrained_net['model.7.bias']

adaptive_norm_net['HR_branch.0.weight'] = pretrained_net['model.2.weight']
adaptive_norm_net['HR_branch.0.bias'] = pretrained_net['model.2.bias']
adaptive_norm_net['HR_branch.3.weight'] = pretrained_net['model.5.weight']
adaptive_norm_net['HR_branch.3.bias'] = pretrained_net['model.5.bias']
adaptive_norm_net['HR_branch.6.weight'] = pretrained_net['model.8.weight']
adaptive_norm_net['HR_branch.6.bias'] = pretrained_net['model.8.bias']
adaptive_norm_net['HR_branch.8.weight'] = pretrained_net['model.10.weight']
adaptive_norm_net['HR_branch.8.bias'] = pretrained_net['model.10.bias']

# for k, v in pretrained_net_degradation.items():
#     adaptive_norm_net[k] = v

print('OK. \n Saving model...')
# torch.save(adaptive_norm_net, '../../experiments/pretrained_models/jpeg_estimation/JPEG80_gray_nonorm_denoise_resnet_DIV2K/jpeg80_964000.pth')
# torch.save(adaptive_norm_net, '../../experiments/pretrained_models/50to15_models/gaussian75_nonorm_denoise_resnet_DIV2K/noise_CNA_adaptive_988000.pth')
# torch.save(adaptive_norm_net, '../../experiments/pretrained_models/basic_model/gaussian_from15to75_resnet_denoise_DIV2K/from15to75_basicmodel_986000.pth')
# torch.save(adaptive_norm_net, '../../experiments/pretrained_models/sr_c16s06/LR_srx4_c16s06_resnet_denoise_DIV2K/c16s06_basicmodel_704000.pth')
torch.save(adaptive_norm_net, '../../experiments/pretrained_models/sr/LR_srx4_resnet_denoise_DIV2K/basicmodel_516000.pth')