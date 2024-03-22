import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm, trange

from data import create_dataloader
from utils.logger import Logger
from models.modules.resnet_architecture.super_mobile_resnet_generator import SuperMobileResnetBlock_with_SPM_bi, BinaryConv2d
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn
import gc


def set_seed(seed):
    cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, task):
        if task == 'train':
            from options.train_options import TrainOptions as Options
            from models import create_model as create_model
        elif task == 'distill':
            from options.distill_options import DistillOptions as Options
            from distillers import create_distiller as create_model
        elif task == 'supernet':
            from options.supernet_options import SupernetOptions as Options
            from supernets import create_supernet as create_model
        else:
            raise NotImplementedError('Unknown task [%s]!!!' % task)
        opt = Options().parse()
        opt.tensorboard_dir = opt.log_dir if opt.tensorboard_dir is None else opt.tensorboard_dir
        print(' '.join(sys.argv))
        if opt.phase != 'train':
            warnings.warn('You are not using training set for %s!!!' % task)
        with open(os.path.join(opt.log_dir, 'opt.txt'), 'a') as f:
            f.write(' '.join(sys.argv) + '\n')
        set_seed(opt.seed)

        dataloader = create_dataloader(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataloader.dataset)  # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        logger = Logger(opt)

        self.opt = opt
        self.dataloader = dataloader
        self.model = model
        self.logger = logger

        warnings.filterwarnings("ignore",
                                message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")

    def evaluate(self, epoch, iter, message):
        start_time = time.time()
        metrics = self.model.evaluate_model(iter)
        wandb.log(metrics)
        self.logger.print_current_metrics(epoch, iter, metrics, time.time() - start_time)
        self.logger.plot(metrics, iter)
        self.logger.print_info(message)
        self.model.save_networks('latest')

    def start(self):
        opt = self.opt
        dataloader = self.dataloader
        model = self.model
        logger = self.logger

        start_epoch = opt.epoch_base
        end_epoch = opt.nepochs + opt.nepochs_decay
        total_iter = opt.iter_base
        epoch_tqdm = trange(start_epoch, end_epoch + 1, desc='Epoch      ', position=0, leave=False)
        self.logger.set_progress_bar(epoch_tqdm)
        for epoch in epoch_tqdm:
            epoch_start_time = time.time()  # timer for entire epoch

            # bi-level setting
            if opt.bi_level_train:
                if epoch < opt.bi_level_start_epoch:
                    opt.no_mac_loss = True
                    opt.no_nuc_loss = True
                    for name, module in model.netG_student.model.named_children():
                        if isinstance(module, SuperMobileResnetBlock_with_SPM_bi):
                            module.mode = 'original'
                            module.pm.weight.requires_grad = False
                            module.spm.weight.requires_grad = False
                    model.netG_student.spm.weight.requires_grad = False
                    model.netG_student.pm1.weight.requires_grad = False
                    model.netG_student.pm2.weight.requires_grad = False
                    model.netG_student.mode = 'original'
                    print('original mode'.center(100, '-'))
                elif opt.bi_level_start_epoch <= epoch <= opt.R_max:  # bi-level optimization
                    if epoch % opt.bi_level_interval == 0:
                        opt.no_mac_loss = False
                        opt.no_nuc_loss = False
                        for name, module in model.netG_student.model.named_children():
                            if isinstance(module, SuperMobileResnetBlock_with_SPM_bi):
                                module.mode = 'prune'
                                module.pm.weight.requires_grad = True
                                module.spm.weight.requires_grad = True
                        model.netG_student.spm.weight.requires_grad = True
                        model.netG_student.pm1.weight.requires_grad = True
                        model.netG_student.pm2.weight.requires_grad = True
                        model.netG_student.mode = 'prune'
                        print('prune mode'.center(100, '-'))
                    else:
                        opt.no_mac_loss = True
                        opt.no_nuc_loss = True
                        for name, module in model.netG_student.model.named_children():
                            if isinstance(module, SuperMobileResnetBlock_with_SPM_bi):
                                module.mode = 'original'
                                module.pm.weight.requires_grad = False
                                module.spm.weight.requires_grad = False
                        model.netG_student.spm.weight.requires_grad = False
                        model.netG_student.pm1.weight.requires_grad = False
                        model.netG_student.pm2.weight.requires_grad = False
                        model.netG_student.mode = 'original'
                        print('original mode'.center(100, '-'))
                elif opt.R_max < epoch:
                    opt.no_mac_loss = True
                    opt.no_nuc_loss = True
                    for name, module in model.netG_student.model.named_children():
                        if isinstance(module, SuperMobileResnetBlock_with_SPM_bi):
                            module.mode = 'original'
                            module.pm.weight.requires_grad = False
                            module.spm.weight.requires_grad = False
                    model.netG_student.spm.weight.requires_grad = False
                    model.netG_student.pm1.weight.requires_grad = False
                    model.netG_student.pm2.weight.requires_grad = False
                    model.netG_student.mode = 'original'
                    print('original mode'.center(100, '-'))
            else:
                if opt.R_max < epoch:
                    opt.no_mac_loss = True
                    opt.no_nuc_loss = True
                    for name, module in model.netG_student.model.named_children():
                        if isinstance(module, SuperMobileResnetBlock_with_SPM_bi):
                            module.mode = 'prune'
                            module.pm.weight.requires_grad = False
                            module.spm.weight.requires_grad = False
                    model.netG_student.spm.weight.requires_grad = False
                    model.netG_student.pm1.weight.requires_grad = False
                    model.netG_student.pm2.weight.requires_grad = False
                    model.netG_student.mode = 'prune'
                    print('freeze mode'.center(100, '-'))

            for i, data_i in enumerate(tqdm(dataloader, desc='Batch      ', position=1, leave=False)):
                iter_start_time = time.time()
                total_iter += 1
                model.set_input(data_i)
                model.optimize_parameters(total_iter)
                wandb.log(model.get_current_losses())

                if total_iter % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    logger.print_current_errors(epoch, total_iter, losses, time.time() - iter_start_time)
                    logger.plot(losses, total_iter)
                    fig = visualize_pruned_model(model.netG_student, total_iter)
                    wandb.log({'pruned structure': wandb.Image(fig)})

                if total_iter % opt.save_latest_freq == 0:
                    self.evaluate(epoch, total_iter,
                                  'Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iter))
                    if model.is_best:
                        model.save_networks('iter%d' % total_iter)

                if opt.scheduler_counter == 'iter':
                    model.update_learning_rate(epoch, total_iter, logger=logger)

                if total_iter % 100 == 0:
                    # CUDA memory manage
                    torch.cuda.empty_cache()
                    gc.collect()

                if total_iter >= opt.niters:
                    break
            logger.print_info(
                'End of epoch %d / %d \t Time Taken: %.2f sec' % (epoch, end_epoch, time.time() - epoch_start_time))
            if epoch % opt.save_epoch_freq == 0 or epoch == end_epoch or total_iter >= opt.niters:
                self.evaluate(epoch, total_iter,
                              'Saving the model at the end of epoch %d, iters %d' % (epoch, total_iter))
                model.save_networks(epoch)
            if opt.scheduler_counter == 'epoch':
                model.update_learning_rate(epoch, total_iter, logger=logger)



def visualize_pruned_model(model, iter):
    dic_model = {'name': [], 'total': [], 'remain': []}

    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            if not 'pm' in name and not 'spm' in name:  # only original conv
                if name == '1':
                    w = model.pm1.weight.detach()
                    binary_w = (w > 0.5).float()
                    pm1 = int(torch.sum(torch.where(binary_w == 1, 1, 0)))
                    dic_model['total'].append(int(module.weight.shape[0]))
                    dic_model['remain'].append(pm1)
                elif name == '4':
                    w = model.pm2.weight.detach()
                    binary_w = (w > 0.5).float()
                    pm2 = int(torch.sum(torch.where(binary_w == 1, 1, 0)))
                    dic_model['total'].append(int(module.weight.shape[0]))
                    dic_model['remain'].append(pm2)
                elif name == '7':
                    w = model.spm.weight.detach()
                    binary_w = (w > 0.5).float()
                    spm = int(torch.sum(torch.where(binary_w == 1, 1, 0)))
                    dic_model['total'].append(int(module.weight.shape[0]))
                    dic_model['remain'].append(spm)
                elif hasattr(model.model[int(name.split('.')[0])], 'pm') and hasattr(model.model[int(name.split('.')[0])], 'spm'):  # SuperMobileResnetBlock
                    w = model.model[int(name.split('.')[0])].pm.weight.detach()
                    binary_w = (w > 0.5).float()
                    pm = int(torch.sum(torch.where(binary_w == 1, 1, 0)))
                    w = model.model[int(name.split('.')[0])].spm.weight.detach()
                    binary_w = (w > 0.5).float()
                    spm = int(torch.sum(torch.where(binary_w == 1, 1, 0)))
                    if '1.conv.2' in name:
                        dic_model['total'].append(int(module.weight.shape[0]))
                        dic_model['remain'].append(pm)
                    elif '6.conv.2' in name:
                        dic_model['total'].append(int(module.weight.shape[0]))
                        dic_model['remain'].append(spm)
                    else:
                        dic_model['total'].append(int(module.weight.shape[0]))
                        dic_model['remain'].append(int(module.weight.shape[0]))
                else:
                    dic_model['total'].append(int(module.weight.shape[0]))
                    dic_model['remain'].append(int(module.weight.shape[0]))
                dic_model['name'].append(name)
    idx = np.arange(len(dic_model['name']))
    bar_width = 0.7
    fig = plt.figure(figsize=(16, 10))
    plt.style.use('seaborn')
    bar_1 = plt.bar(idx, dic_model['total'], bar_width, color='gray', alpha=0.5)
    plt.bar(idx, dic_model['remain'], bar_width, color='green', alpha=0.5)
    plt.xticks(list(idx), dic_model['name'], rotation=90)
    plt.ylim(0, 150)
    plt.title(f'Cyclegan (iter : {iter})', size=15, weight='bold')
    plt.ylabel('output channels', size=20, weight='bold')
    plt.xlabel('index', size=15, weight='bold')

    for i, rect in enumerate(bar_1):
        remain_ratio = int(dic_model['remain'][i] / dic_model['total'][i] * 100)
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height + 2, f'{remain_ratio}%', ha='center', va='bottom',
                 size=10, rotation=90)

    plt.grid(True)
    plt.tight_layout()
    return fig