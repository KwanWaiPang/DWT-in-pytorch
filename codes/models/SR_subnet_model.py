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
        self.subnet = networks.define_sub(opt).to(self.device)
        self.load()

        if self.is_train:
            self.netG.train()
            self.subnet.train()
            # self.subnet.eval()

            # loss
            loss_type_noise = train_opt['pixel_criterion_noise']
            if loss_type_noise == 'l1':
                self.cri_pix_noise = nn.L1Loss().to(self.device)
            elif loss_type_noise == 'l2':
                self.cri_pix_noise = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type_noise))
            self.l_pix_noise_w = train_opt['pixel_weight_noise']

            loss_reg_noise = train_opt['pixel_criterion_reg_noise']
            if loss_reg_noise == 'tv':
                self.cri_pix_reg_noise = TVLoss(0.00001).to(self.device)

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

    def feed_data(self, data, need_HR=True):
        self.var_L = data['LR'].to(self.device)  # LR
        self.real_H = data['HR'].to(self.device)  # HR
        self.mid_L = data['MR'].to(self.device)  # MR

    def __define_grad_params(self, finetune_type=None):

        optim_params = []

        if finetune_type == 'sft':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('Gate') >= 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
        elif finetune_type == 'sub_sft':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('Gate') >= 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
            for k, v in self.subnet.named_parameters():  # can optimize for a part of the model
                v.requires_grad = False
                if k.find('degration') >= 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
        elif finetune_type == 'basic' or finetune_type == 'sft_basic':
            for k, v in self.netG.named_parameters():
                v.requires_grad = True
                optim_params.append(v)
                print('we only optimize params: {}'.format(k))
            for k, v in self.subnet.named_parameters():
                v.requires_grad = False
        else:
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                    print('params [{:s}] will optimize.'.format(k))
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            for k, v in self.subnet.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                    print('params [{:s}] will optimize.'.format(k))
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
        return optim_params

    def optimize_parameters(self, step):

        self.optimizer_G.zero_grad()

        self.fake_noise = self.subnet(self.var_L)
        l_pix_noise = self.l_pix_noise_w * self.cri_pix_noise(self.fake_noise, self.mid_L)
        l_pix_noise = l_pix_noise + self.cri_pix_reg_noise(self.fake_noise)
        # self.fake_H = self.netG(torch.cat((self.var_L, self.fake_noise), 1))
        self.fake_H = self.netG((self.var_L, self.fake_noise))
        l_pix_basic = self.l_pix_basic_w * self.cri_pix_basic(self.fake_H, self.real_H)
        l_pix = l_pix_noise + l_pix_basic
        l_pix.backward()

        # self.fake_noise = self.subnet(self.var_L)
        # # self.fake_H = self.netG(torch.cat((self.var_L, self.fake_noise), 1))
        # self.fake_H = self.netG((self.var_L, self.fake_noise))
        # l_pix = self.l_pix_basic_w * self.cri_pix_basic(self.fake_H, self.real_H)
        # l_pix.backward()

        self.optimizer_G.step()

        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        self.subnet.eval()
        if self.is_train:
            for v in self.optim_params:
                v.requires_grad = False
        else:
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
            for k, v in self.subnet.named_parameters():
                v.requires_grad = False
        self.fake_noise = self.subnet(self.var_L)
        # self.fake_H = self.netG(torch.cat((self.var_L, self.fake_noise), 1))
        self.fake_H = self.netG((self.var_L, self.fake_noise))
        if self.is_train:
            for v in self.optim_params:
                v.requires_grad = True
        else:
            for k, v in self.netG.named_parameters():
                v.requires_grad = True
            for k, v in self.subnet.named_parameters():
                v.requires_grad = True
        self.netG.train()
        self.subnet.train()
        # self.subnet.eval()

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

            # subnet
            s, n = self.get_network_description(self.subnet)
            print('Number of parameters in subnet: {:,d}'.format(n))
            message = '\n\n\n-------------- subnet --------------\n' + s + '\n'
            with open(network_path, 'a') as f:
                f.write(message)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        load_path_sub = self.opt['path']['pretrain_model_sub']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        if load_path_sub is not None:
            print('loading model for subnet [{:s}] ...'.format(load_path_sub))
            self.load_network(load_path_sub, self.subnet)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.subnet, 'sub', iter_label)
