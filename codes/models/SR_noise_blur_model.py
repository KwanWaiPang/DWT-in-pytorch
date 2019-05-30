import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from .modules.loss import TVLoss


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        train_opt = opt['train']
        finetune_type = opt['finetune_type']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.subnet_noise = networks.define_sub(opt).to(self.device)
        self.subnet_blur = networks.define_sub2(opt).to(self.device)
        self.load()

        if self.is_train:
            self.netG.train()
            if finetune_type in ['basic', 'sft_basic', 'sft', 'sub_sft']:
                self.subnet_noise.eval()
                self.subnet_blur.eval()
            else:
                self.subnet_noise.train()
                self.subnet_blur.train()

            # loss on noise
            loss_type_noise = train_opt['pixel_criterion_noise']
            if loss_type_noise == 'l1':
                self.cri_pix_noise = nn.L1Loss().to(self.device)
            elif loss_type_noise == 'l2':
                self.cri_pix_noise = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Noise loss type [{:s}] is not recognized.'.format(loss_type_noise))
            self.l_pix_noise_w = train_opt['pixel_weight_noise']

            loss_reg_noise = train_opt['pixel_criterion_reg_noise']
            if loss_reg_noise == 'tv':
                self.cri_pix_reg_noise = TVLoss(0.00001).to(self.device)

            # loss on blur
            loss_type_blur = train_opt['pixel_criterion_blur']
            if loss_type_blur == 'l1':
                self.cri_pix_blur = nn.L1Loss().to(self.device)
            elif loss_type_blur == 'l2':
                self.cri_pix_blur = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Blur loss type [{:s}] is not recognized.'.format(loss_type_blur))
            self.l_pix_blur_w = train_opt['pixel_weight_blur']

            loss_reg_blur = train_opt['pixel_criterion_reg_blur']
            if loss_reg_blur == 'tv':
                self.cri_pix_reg_blur = TVLoss(0.00001).to(self.device)

            loss_type_basic = train_opt['pixel_criterion_basic']
            if loss_type_basic == 'l1':
                self.cri_pix_basic = nn.L1Loss().to(self.device)
            elif loss_type_basic == 'l2':
                self.cri_pix_basic = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type_basic))
            self.l_pix_basic_w = train_opt['pixel_weight_basic']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0

            self.optim_params = self.__define_grad_params(finetune_type)

            self.optimizer_G = torch.optim.Adam(
                self.optim_params, lr=train_opt['lr_G'], weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_MR=True, need_MAT=True):
        self.var_L = data['LR'].to(self.device)  # LR
        self.real_H = data['HR'].to(self.device)  # HR
        if need_MR:
            self.mid_L = data['MR'].to(self.device)  # MR
        if need_MAT:
            self.real_blur = data['MAT'].to(self.device)

    def __define_grad_params(self, finetune_type=None):

        optim_params = []

        if finetune_type == 'sft':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('Gate') >= 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
        else:
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                    print('params [{:s}] will optimize.'.format(k))
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            for k, v in self.subnet_noise.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                    print('params [{:s}] will optimize.'.format(k))
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            for k, v in self.subnet_blur.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                    print('params [{:s}] will optimize.'.format(k))
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
        return optim_params

    def optimize_parameters(self, step):

        self.optimizer_G.zero_grad()

        self.fake_noise = self.subnet_noise(self.var_L)
        l_pix_noise = self.l_pix_noise_w * self.cri_pix_noise(self.fake_noise, self.mid_L)
        l_pix_noise = l_pix_noise + self.cri_pix_reg_noise(self.fake_noise)

        input_noise = torch.cat((self.var_L, self.fake_noise), 1)
        self.fake_blur = self.subnet_blur(input_noise)
        l_pix_blur = self.l_pix_blur_w * self.cri_pix_blur(self.fake_blur*16, self.real_blur)
        l_pix_blur = l_pix_blur + self.cri_pix_reg_blur(self.fake_blur)

        input_noise_blur = torch.cat((input_noise, self.fake_blur), 1)
        self.fake_H = self.netG(input_noise_blur)
        l_pix_basic = self.l_pix_basic_w * self.cri_pix_basic(self.fake_H, self.real_H)
        l_pix = l_pix_noise + l_pix_blur + l_pix_basic
        l_pix.backward()

        self.optimizer_G.step()

        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        self.subnet_noise.eval()
        self.subnet_blur.eval()
        if self.is_train:
            for v in self.optim_params:
                v.requires_grad = False
        else:
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
            for k, v in self.subnet_noise.named_parameters():
                v.requires_grad = False
            for k, v in self.subnet_blur.named_parameters():
                v.requires_grad = False
        self.fake_noise = self.subnet_noise(self.var_L)
        input_noise = torch.cat((self.var_L, self.fake_noise), 1)
        self.fake_blur = self.subnet_blur(input_noise)
        input_noise_blur = torch.cat((input_noise, self.fake_blur), 1)
        self.fake_H = self.netG(input_noise_blur)
        if self.is_train:
            for v in self.optim_params:
                v.requires_grad = True
        else:
            for k, v in self.netG.named_parameters():
                v.requires_grad = True
            for k, v in self.subnet_noise.named_parameters():
                v.requires_grad = True
            for k, v in self.subnet_blur.named_parameters():
                v.requires_grad = True
        self.netG.train()
        if self.opt['finetune_type'] in ['basic', 'sft_basic', 'sft', 'sub_sft']:
            self.subnet_noise.eval()
            self.subnet_blur.eval()
        else:
            self.subnet_noise.train()
            self.subnet_blur.eval()

    # def test(self):
    #     self.netG.eval()
    #     for k, v in self.netG.named_parameters():
    #         v.requires_grad = False
    #     self.fake_H = self.netG(self.var_L)
    #     for k, v in self.netG.named_parameters():
    #         v.requires_grad = True
    #     self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['MR'] = self.fake_noise.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # G
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

            # noise subnet
            s, n = self.get_network_description(self.subnet_noise)
            print('Number of parameters in noise subnet: {:,d}'.format(n))
            message = '\n\n\n-------------- noise subnet --------------\n' + s + '\n'
            with open(network_path, 'a') as f:
                f.write(message)

            # blur subnet
            s, n = self.get_network_description(self.subnet_blur)
            print('Number of parameters in blur subnet: {:,d}'.format(n))
            message = '\n\n\n-------------- blur subnet --------------\n' + s + '\n'
            with open(network_path, 'a') as f:
                f.write(message)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        load_path_sub_noise = self.opt['path']['pretrain_model_sub_noise']
        load_path_sub_blur = self.opt['path']['pretrain_model_sub_blur']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        if load_path_sub_noise is not None:
            print('loading model for noise subnet [{:s}] ...'.format(load_path_sub_noise))
            self.load_network(load_path_sub_noise, self.subnet_noise)
        if load_path_sub_blur is not None:
            print('loading model for blur subnet [{:s}] ...'.format(load_path_sub_blur))
            self.load_network(load_path_sub_blur, self.subnet_blur)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.subnet_noise, 'sub_noise', iter_label)
        self.save_network(self.save_dir, self.subnet_blur, 'sub_blur', iter_label)
