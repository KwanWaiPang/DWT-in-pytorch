import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
from scipy.io import loadmat


class LRMRMATHRDataset(data.Dataset):
    '''
    Read LR, MR and HR image pair.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRMRMATHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_MR = None
        self.paths_HR = None
        self.paths_MAT = None
        self.LR_env = None  # environment for lmdb
        self.MR_env = None
        self.HR_env = None
        self.MAT_env = None

        self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
        self.MR_env, self.paths_MR = util.get_image_paths(opt['data_type'], opt['dataroot_MR'])
        self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])
        self.MAT_env, self.paths_MAT = util.get_image_paths(opt['data_type'], opt['dataroot_MAT'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_MR:
            assert len(self.paths_LR) == len(self.paths_MR), \
                'MR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_MR))

        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path, MR_path, MAT_path = None, None, None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']
        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)

        # # modcrop in the validation / test phase
        # if self.opt['phase'] != 'train':
        #     img_HR = util.modcrop(img_HR, scale)

        LR_path = self.paths_LR[index]
        img_LR = util.read_img(self.LR_env, LR_path)

        MR_path = self.paths_MR[index]
        img_MR = util.read_img(self.MR_env, MR_path)

        # get mat file
        MAT_path = self.paths_MAT[index]
        img_MAT = loadmat(MAT_path)['im_residual']
        # kernel_gt = loadmat(MAT_path)['kernel_gt']

        # img_MAT = np.zeros_like(img_LR)

        if self.opt['noise_gt']:
            img_MR = img_LR - img_MR

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, C = img_LR.shape
            LR_size = HR_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_MR = img_MR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            img_MAT = img_MAT[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

            # for ind, value in enumerate(kernel_gt):
            #     img_MAT[:, :, ind] = np.tile(value, (LR_size, LR_size))

            # augmentation - flip, rotate
            img_MR, img_MAT, img_LR, img_HR = util.augment([img_MR, img_MAT, img_LR, img_HR], self.opt['use_flip'], \
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
            img_MR = img_MR[:, :, [2, 1, 0]]
            img_MAT = img_MAT[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        img_MR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_MR, (2, 0, 1)))).float()
        img_MAT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_MAT, (2, 0, 1)))).float()

        return {'HR': img_HR, 'LR': img_LR, 'MR': img_MR, 'MAT': img_MAT, 'HR_path': HR_path, 'MR_path': MR_path,
                'LR_path': LR_path, 'MAT_path': MAT_path}

    def __len__(self):
        return len(self.paths_HR)
