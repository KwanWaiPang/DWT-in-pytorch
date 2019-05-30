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
        res_mode = opt['network_G']['mode']
        net_type = opt['network_G']['which_model_G']
        finetune_type = opt['finetune_type']
        init_norm_type = opt['init_norm_type']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.load()

        if init_norm_type is not None:
            self.__init_norm(res_mode, net_type, init_norm_type)

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            loss_reg = train_opt['pixel_criterion_reg']
            if loss_reg == 'tv':
                self.cri_pix_reg = TVLoss(0.00001).to(self.device)

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            #
            self.optim_params = self.__define_grad_params(finetune_type, res_mode, net_type)

            self.optimizer_G = torch.optim.Adam(
                self.optim_params, lr=train_opt['lr_G'], weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)

            # optim_params_first_layer = []
            # optim_params_other = []
            # for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            #     if 'model.0' in k:
            #         optim_params_first_layer.append(v)
            #     else:
            #         optim_params_other.append(v)
            # self.optimizer_G_first_layer = torch.optim.Adam(optim_params_first_layer, lr=train_opt['lr_G'] * 10,
            #                                                 weight_decay=wd_G)
            # self.optimizer_G_other = torch.optim.Adam(optim_params_other, lr=train_opt['lr_G'], weight_decay=wd_G)
            # self.optimizers.append(self.optimizer_G_first_layer)
            # self.optimizers.append(self.optimizer_G_other)

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

    def __define_grad_params(self, finetune_type=None, res_mode='CNA', net_type='denoise_resnet'):

        optim_params = []
        if finetune_type == 'norm':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                # if net_type == 'arcnn' or net_type == 'srcnn':
                #     if k.find('model.1') >= 0:
                #         v.requires_grad = True
                #         optim_params.append(v)
                #         print('we only optimize params: {}'.format(k))
                if res_mode == "CNA" or res_mode == "NCA":
                    # if k.find('res.1') >= 0 or k.find('res.4') >= 0 or k.find("sub.17") >= 0:
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        optim_params.append(v)
                        print('we only optimize params: {}'.format(k))
        elif finetune_type == 'estimation':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('Gate') >= 0 or k.find('degration') >= 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
        elif finetune_type == 'sft':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('Gate') >= 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
        elif finetune_type == 'fea_upsample':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('res') < 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
        elif finetune_type == 'upsample':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('model.1') < 0 and k.find('model.0') < 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
        elif finetune_type == 'first_layer':
            for k, v in self.netG.named_parameters():
                v.requires_grad = False
                if k.find('model.0') >= 0:
                    v.requires_grad = True
                    optim_params.append(v)
                    print('we only optimize params: {}'.format(k))
        else:
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [%s] will not optimize.' % k)
        return optim_params

    def __init_norm(self, res_mode='CNA', net_type='denoise_resnet', init_norm_type='zero'):
        for k, v in self.netG.named_parameters():
            # if net_type == "arcnn" or net_type == "srcnn":
            #     if k.find('model.1') >= 0:
            #         if init_norm_type == 'zero':
            #             print(k, 'initialize with 0')
            #             nn.init.constant(v, 0)
            if res_mode == "CNA" or res_mode == "NCA":
                # if k.find('res.1') >= 0 or k.find('res.4') >= 0 or k.find("sub.17") >= 0:
                if k.find('transformer') >= 0:
                    if init_norm_type == "instance" or init_norm_type == "batch":
                        if "weight" in k:
                            print(k, "initialize with xavier_uniform")
                            nn.init.constant(v, 1)
                        elif "bias" in k:
                            print(k, "initialize with 0")
                            nn.init.constant(v, 0)
                    elif init_norm_type == 'zero':
                        print(k, "initialize with 0")
                        nn.init.constant(v, 0)

    def feed_data(self, data, need_HR=True, noise_gt=False):
        self.var_L = data['LR'].to(self.device)  # LR
        if need_HR:
            self.real_H = data['HR'].to(self.device)  # HR

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        # l_pix = l_pix + self.cri_pix_reg(self.fake_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def test_sigma(self):
        # noise level estimation: sigma
        self.netG.eval()
        with torch.no_grad():
            sigma = self.netG(self.var_L)
        self.netG.train()
        return sigma

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()
        for k, v in self.netG.named_parameters():
            v.requires_grad = False

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)

        for k, v in self.netG.named_parameters():
            v.requires_grad = True
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            with open(network_path, 'w') as f:
                f.write(message)

    def update(self, new_model_dict):
        if isinstance(self.netG, nn.DataParallel):
            network = self.netG.module
            network.load_state_dict(new_model_dict)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
