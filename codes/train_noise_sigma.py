import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.logger import Logger, PrintLogger


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)

    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiments if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and \
        not key == 'pretrain_model_G' and not key == 'pretrain_model_D'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # print to file and std_out simultaneously
    sys.stdout = PrintLogger(opt['path']['log'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    val_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epoches = int(math.ceil(total_iters / train_size))
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif 'val' in phase:
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
            val_loaders.append(val_loader)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    # create logger
    logger = Logger(opt)


    current_step = 0
    start_time = time.time()
    print('---------- Start training -------------')
    for epoch in range(total_epoches):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            # level = random.randint(0, 80)
            level = random.uniform(0, 80)
            # train_data = add_Gaussian_noise(train_data, level)
            train_data = add_dependent_noise(train_data, level)
            model.feed_data(train_data)

            # training
            model.optimize_parameters(current_step)

            time_elapsed = time.time() - start_time
            start_time = time.time()

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                for k, v in logs.items():
                    print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt)

            # save models
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                print('Saving the model at the end of iter {:d}.'.format(current_step))
                model.save(current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                print('---------- validation -------------')
                for val_loader in val_loaders:
                    val_set_name = val_loader.dataset.opt['name']
                    print('validation [%s]...' % val_set_name)

                    print_rlt = OrderedDict()

                    start_time = time.time()
                    avg_l2 = 0.0
                    idx = 0

                    sigma_gt = int(val_set_name[-2:])

                    sigma_pre = []

                    for val_data in val_loader:
                        idx += 1
                        val_data = add_dependent_noise(val_data, sigma_gt)
                        model.feed_data(val_data, need_HR=False)
                        sigma = model.test_sigma().squeeze().float().cpu().item()
                        sigma_pre.append(sigma)
                        avg_l2 += (sigma - sigma_gt)**2

                    print('sigma: {}'.format(sigma_pre))

                    # log the sigma, time for each quality
                    time_elapsed = time.time() - start_time

                    log_val_name = 'l2_noise{}'.format(sigma_gt)
                    print_rlt[log_val_name] = avg_l2

                    print_rlt['time'] = time_elapsed

                    # Save to log
                    print_rlt['model'] = opt['model']
                    print_rlt['epoch'] = epoch
                    print_rlt['iters'] = current_step
                    logger.print_format_results('val', print_rlt)
                print('-----------------------------------')
                # end of the validation

            # update learning rate
            model.update_learning_rate()

    print('Saving the final model.')
    model.save('latest')
    print('End of training.')


def add_Gaussian_noise(data, level):
    # data_HR_batch = data['HR']
    # noise = torch.FloatTensor(data_HR_batch.size()).normal_(mean=0, std=level/255.)
    data_LR_batch = data['LR']
    noise = torch.randn(data_LR_batch.size()) * (level/255.)
    data['LR'] = data_LR_batch + noise
    # range 0. ~ 1.
    data['LR'].clamp_(0., 1.)
    level_gt = torch.Tensor(16, 1, 1, 1).fill_(level).float()
    data['HR'] = level_gt
    return data


def add_dependent_noise(data, level):
    # data_HR_batch = data['HR']
    # noise = torch.FloatTensor(data_HR_batch.size()).normal_(mean=0, std=level/255.)
    data_LR_batch = data['LR']
    noise_s_map = data_LR_batch * (level / 255.)
    noise_s = torch.randn(data_LR_batch.size()) * noise_s_map
    # noise = torch.randn(data_LR_batch.size()) * (level/255.)
    data['LR'] = data_LR_batch + noise_s
    # range 0. ~ 1.
    data['LR'].clamp_(0., 1.)
    level_gt = torch.Tensor(16, 1, 1, 1).fill_(level).float()
    data['HR'] = level_gt
    return data

def add_mix_noise(data, level_add, level_times):
    # data_HR_batch = data['HR']
    # noise = torch.FloatTensor(data_HR_batch.size()).normal_(mean=0, std=level/255.)
    data_LR_batch = data['LR']
    noise_add = torch.randn(data_LR_batch.size()) * (level_add/255.)
    noise_times = torch.randn(data_LR_batch.size()) * (level_times/255.)
    data['LR'] = data_LR_batch * noise_times + noise_add
    # range 0. ~ 1.
    data['LR'].clamp_(0., 1.)
    return data

if __name__ == '__main__':
    # # OpenCV get stuck in transform when used in DataLoader
    # # https://github.com/pytorch/pytorch/issues/1838
    # # However, cause problem reading lmdb
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()
