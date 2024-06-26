import itertools
import os

import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel

import models.modules.loss
from data import create_eval_dataloader
from metric import create_metric_models
from models import networks
from models.base_model import BaseModel
from models.modules.loss import GANLoss
from models.modules.super_modules import SuperConv2d
from utils import util
from argparse import ArgumentParser
from models.modules.resnet_architecture.super_mobile_resnet_generator import BinaryConv2d
from models.modules.loss import append_loss_mac, append_loss_nuc
import wandb

class BaseResnetDistiller(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(BaseResnetDistiller, BaseResnetDistiller).modify_commandline_options(parser, is_train)
        assert isinstance(parser, ArgumentParser)
        parser.add_argument('--recon_loss_type', type=str, default='l1',
                            choices=['l1', 'l2', 'smooth_l1', 'vgg'],
                            help='the type of the reconstruction loss')
        parser.add_argument('--lambda_distill', type=float, default=1,
                            help='weights for the intermediate activation distillation loss')
        parser.add_argument('--lambda_recon', type=float, default=100,
                            help='weights for the reconstruction loss.')
        parser.add_argument('--lambda_gan', type=float, default=1,
                            help='weight for gan loss')
        parser.add_argument('--teacher_dropout_rate', type=float, default=0)
        parser.add_argument('--student_dropout_rate', type=float, default=0)
        parser.add_argument('--no_mac_loss', action="store_true", help='turn off mac loss')
        parser.add_argument('--target_ratio', type=float, default=0.5)
        parser.add_argument('--alpha_mac', type=float, default=0.5)
        parser.add_argument('--mac_front', action="store_true", help='mac loss for front')
        parser.add_argument('--mac_downsample', action="store_true", help='mac loss for downsampling')
        parser.add_argument('--mac_resnet', action="store_true", help='mac loss for resnet')
        parser.add_argument('--mac_upsample', action="store_true", help='mac loss for upsampling')
        parser.add_argument('--no_nuc_loss', action="store_true", help='turn off nuc loss')
        parser.add_argument('--alpha_nuc', type=float, default=0.001)
        parser.add_argument('--R_max', type=int, default=400)
        parser.add_argument('--proj_name', type=str, default='test')
        parser.add_argument('--log_name', type=str, default='test')
        parser.add_argument("--bi_level_train", action="store_true", help="turn on bi level train")
        parser.add_argument("--bi_level_interval", default="1", type=int, help="pruning frequency")
        parser.add_argument("--bi_level_start_epoch", default="10", type=int, help="train original model until bi_level_start_epoch")
        parser.set_defaults(teacher_netG='mobile_resnet_9blocks', teacher_ngf=64,
                            student_netG='mobile_resnet_9blocks', student_ngf=48)
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        valid_netGs = ['resnet_9blocks', 'mobile_resnet_9blocks',
                       'super_mobile_resnet_9blocks', 'sub_mobile_resnet_9blocks', 'super_mobile_resnet_9blocks_SPM_bi']
        assert opt.teacher_netG in valid_netGs and opt.student_netG in valid_netGs
        super(BaseResnetDistiller, self).__init__(opt)
        self.loss_names = ['G_gan', 'G_distill', 'G_recon', 'D_fake', 'D_real', 'netG_student_mac', 'netG_student_nuc', 'netG_student_mac_front', 'netG_student_mac_downsample', 'netG_student_mac_resnet', 'netG_student_mac_upsample']
        self.optimizers = []
        self.image_paths = []
        self.visual_names = ['real_A', 'Sfake_B', 'Tfake_B', 'real_B']
        self.model_names = ['netG_student', 'netG_teacher', 'netD']
        self.netG_teacher = networks.define_G(opt.teacher_netG, input_nc=opt.input_nc,
                                              output_nc=opt.output_nc, ngf=opt.teacher_ngf,
                                              norm=opt.norm, dropout_rate=opt.teacher_dropout_rate,
                                              gpu_ids=self.gpu_ids, opt=opt)
        self.netG_student = networks.define_G(opt.student_netG, input_nc=opt.input_nc,
                                              output_nc=opt.output_nc, ngf=opt.student_ngf,
                                              norm=opt.norm, dropout_rate=opt.student_dropout_rate,
                                              init_type=opt.init_type, init_gain=opt.init_gain,
                                              gpu_ids=self.gpu_ids, opt=opt)
        # initialize PM, SPM weight to 0.6
        for name, module in self.netG_student.model.named_modules():
            if isinstance(module, BinaryConv2d):
                nn.init.constant_(module.weight, 0.6)
        nn.init.constant_(self.netG_student.pm1.weight, 0.6)
        nn.init.constant_(self.netG_student.pm2.weight, 0.6)
        nn.init.constant_(self.netG_student.spm.weight, 0.6)
        nn.init.constant_(self.netG_student.pm3.weight, 0.6)
        nn.init.constant_(self.netG_student.pm4.weight, 0.6)


        if hasattr(opt, 'distiller'):
            self.netG_pretrained = networks.define_G(opt.pretrained_netG, input_nc=opt.input_nc,
                                                     output_nc=opt.output_nc, ngf=opt.pretrained_ngf,
                                                     norm=opt.norm, gpu_ids=self.gpu_ids, opt=opt)
        if opt.dataset_mode == 'aligned':
            self.netD = networks.define_D(opt.netD, input_nc=opt.input_nc + opt.output_nc,
                                          ndf=opt.ndf, n_layers_D=opt.n_layers_D, norm=opt.norm,
                                          init_type=opt.init_type, init_gain=opt.init_gain,
                                          gpu_ids=self.gpu_ids, opt=opt)
        elif opt.dataset_mode == 'unaligned':
            self.netD = networks.define_D(opt.netD, input_nc=opt.output_nc,
                                          ndf=opt.ndf, n_layers_D=opt.n_layers_D, norm=opt.norm,
                                          init_type=opt.init_type, init_gain=opt.init_gain,
                                          gpu_ids=self.gpu_ids, opt=opt)
        else:
            raise NotImplementedError('Unknown dataset mode [%s]!!!' % opt.dataset_mode)

        self.netG_teacher.eval()
        self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
        if opt.recon_loss_type == 'l1':
            self.criterionRecon = torch.nn.L1Loss()
        elif opt.recon_loss_type == 'l2':
            self.criterionRecon = torch.nn.MSELoss()
        elif opt.recon_loss_type == 'smooth_l1':
            self.criterionRecon = torch.nn.SmoothL1Loss()
        elif opt.recon_loss_type == 'vgg':
            self.criterionRecon = models.modules.loss.VGGLoss(self.device)
        else:
            raise NotImplementedError('Unknown reconstruction loss type [%s]!' % opt.loss_type)

        if isinstance(self.netG_teacher, nn.DataParallel):
            self.mapping_layers = ['module.model.%d' % i for i in range(9, 21, 3)]
        else:
            self.mapping_layers = ['model.%d' % i for i in range(9, 21, 3)]

        self.netAs = []
        self.Tacts, self.Sacts = {}, {}

        param_groups = [
            {"params": [], "lr": opt.lr * 0.1, "betas": (opt.beta1, 0.999)},  # SPM learning rate * 0.1
            {"params": [], "lr": opt.lr, "betas": (opt.beta1, 0.999)}
        ]

        for name, param in self.netG_student.named_parameters():
            if 'spm' in name:
                param_groups[0]["params"].append(param)
            else:
                param_groups[1]["params"].append(param)
        # G_params = [self.netG_student.parameters()]
        # G_params = param_groups
        for i, n in enumerate(self.mapping_layers):
            ft, fs = self.opt.teacher_ngf, self.opt.student_ngf
            if hasattr(opt, 'distiller'):
                netA = nn.Conv2d(in_channels=fs * 4, out_channels=ft * 4, kernel_size=1). \
                    to(self.device)
            else:
                netA = SuperConv2d(in_channels=fs * 4, out_channels=ft * 4, kernel_size=1). \
                    to(self.device)
            networks.init_net(netA)
            for name, param in netA.named_parameters():
                param_groups[1]["params"].append(param)
            # G_params.append(list(netA.parameters()))
            self.netAs.append(netA)
            self.loss_names.append('G_distill%d' % i)

        # self.optimizer_G = torch.optim.Adam(itertools.chain(*G_params), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(param_groups)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.eval_dataloader = create_eval_dataloader(self.opt, direction=opt.direction)
        self.inception_model, self.drn_model, _ = create_metric_models(opt, device=self.device)
        self.npz = np.load(opt.real_stat_path)
        self.is_best = False

        if opt.no_mac_loss:
            opt.alpha_mac = 0
        if opt.no_nuc_loss:
            opt.alpha_nuc = 0

        wandb.init(project=opt.proj_name)
        wandb.run.name = opt.log_dir.split('/')[-1]
        wandb.run.save()
        wandb.config.update(vars(opt))
        print('wandb init')


    def setup(self, opt, verbose=True):
        super(BaseResnetDistiller, self).setup(opt, verbose)
        if self.opt.lambda_distill > 0:
            def get_activation(mem, name):
                def get_output_hook(module, input, output):
                    mem[name + str(output.device)] = output

                return get_output_hook

            def add_hook(net, mem, mapping_layers):
                for n, m in net.named_modules():
                    if n in mapping_layers:
                        m.register_forward_hook(get_activation(mem, n))

            add_hook(self.netG_teacher, self.Tacts, self.mapping_layers)
            add_hook(self.netG_student, self.Sacts, self.mapping_layers)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_single_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        raise NotImplementedError

    def backward_D(self):
        if self.opt.dataset_mode == 'aligned':
            fake = torch.cat((self.real_A, self.Sfake_B), 1).detach()
            real = torch.cat((self.real_A, self.real_B), 1).detach()
        else:
            fake = self.Sfake_B.detach()
            real = self.real_B.detach()

        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)

        pred_real = self.netD(real)
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def calc_distill_loss(self):
        raise NotImplementedError

    def backward_G(self):
        wandb.log({'lr/spm': self.optimizers[0].param_groups[0]['lr'], 'lr/rest': self.optimizers[0].param_groups[1]['lr']})
        if self.opt.dataset_mode == 'aligned':
            self.loss_G_recon = self.criterionRecon(self.Sfake_B, self.real_B) * self.opt.lambda_recon
            fake = torch.cat((self.real_A, self.Sfake_B), 1)
        else:
            self.loss_G_recon = self.criterionRecon(self.Sfake_B, self.Tfake_B) * self.opt.lambda_recon
            fake = self.Sfake_B
        pred_fake = self.netD(fake)
        self.loss_G_gan = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.lambda_gan
        if self.opt.lambda_distill > 0:
            self.loss_G_distill = self.calc_distill_loss() * self.opt.lambda_distill
        else:
            self.loss_G_distill = 0
        if not self.opt.no_mac_loss:
            cur_macs_front, remain_in_nc = self.netG_student.get_macs_front()
            cur_macs_downsample, remain_in_nc = self.netG_student.get_macs_downsample(remain_in_nc)
            cur_macs_resnet, remain_in_nc = self.netG_student.get_macs_resnet(remain_in_nc)
            cur_macs_upsample, remain_in_nc = self.netG_student.get_macs_upsample(remain_in_nc)
            if self.netG_student.pm1.in_channels == 32:
                target_macs_front = torch.tensor([0.3083]).cuda() * (1 - self.opt.target_ratio)
                target_macs_downsample = torch.tensor([0.6040]).cuda() * (1 - self.opt.target_ratio)
                target_macs_resnet = torch.tensor([1.2929]).cuda() * (1 - self.opt.target_ratio)
                target_macs_upsample = torch.tensor([2.4159]).cuda() * (1 - self.opt.target_ratio)
            elif self.netG_student.pm1.in_channels == 64:
                target_macs_front = torch.tensor([0.6166]).cuda() * (1 - self.opt.target_ratio)
                target_macs_downsample = torch.tensor([2.4159]).cuda() * (1 - self.opt.target_ratio)
                target_macs_resnet = torch.tensor([5.0017]).cuda() * (1 - self.opt.target_ratio)
                target_macs_upsample = torch.tensor([9.6637]).cuda() * (1 - self.opt.target_ratio)
            self.loss_netG_student_mac_front = append_loss_mac(cur_macs_front, target_macs_front, self.opt.alpha_mac)
            self.loss_netG_student_mac_downsample = append_loss_mac(cur_macs_downsample, target_macs_downsample, self.opt.alpha_mac)
            self.loss_netG_student_mac_resnet = append_loss_mac(cur_macs_resnet, target_macs_resnet, self.opt.alpha_mac)
            self.loss_netG_student_mac_upsample = append_loss_mac(cur_macs_upsample, target_macs_upsample, self.opt.alpha_mac)

            self.loss_netG_student_mac = torch.tensor([0.]).cuda()
            if self.opt.mac_front:
                self.loss_netG_student_mac += self.loss_netG_student_mac_front
            if self.opt.mac_downsample:
                self.loss_netG_student_mac += self.loss_netG_student_mac_downsample
            if self.opt.mac_resnet:
                self.loss_netG_student_mac += self.loss_netG_student_mac_resnet
            if self.opt.mac_upsample:
                self.loss_netG_student_mac += self.loss_netG_student_mac_upsample

            wandb.log({'cur macs front' : cur_macs_front})
            wandb.log({'cur macs downsample' : cur_macs_downsample})
            wandb.log({'cur macs resnet' : cur_macs_resnet})
            wandb.log({'cur macs upsample' : cur_macs_upsample})
            wandb.log({'cur macs' : cur_macs_front + cur_macs_downsample + cur_macs_resnet + cur_macs_upsample})
            if self.netG_student.pm1.in_channels == 32:
                wandb.log({'cur macs front ratio': cur_macs_front / 0.3083})
                wandb.log({'cur macs downsample ratio': cur_macs_downsample / 0.6040})
                wandb.log({'cur macs resnet ratio': cur_macs_resnet / 1.2929})
                wandb.log({'cur macs upsample ratio': cur_macs_upsample / 2.4159})
                wandb.log({'cur macs ratio' : (cur_macs_front + cur_macs_downsample + cur_macs_resnet + cur_macs_upsample) / 4.6211})
            elif self.netG_student.pm1.in_channels == 64:
                wandb.log({'cur macs front ratio': cur_macs_front / 0.6166})
                wandb.log({'cur macs downsample ratio': cur_macs_downsample / 2.4159})
                wandb.log({'cur macs resnet ratio': cur_macs_resnet / 5.0017})
                wandb.log({'cur macs upsample ratio': cur_macs_upsample / 9.6637})
                wandb.log({'cur macs ratio': (cur_macs_front + cur_macs_downsample + cur_macs_resnet + cur_macs_upsample) / 17.6979})
            self.cur_macs = cur_macs_front + cur_macs_downsample + cur_macs_resnet + cur_macs_upsample
            del cur_macs_front, cur_macs_downsample, cur_macs_resnet, cur_macs_upsample
            del target_macs_front, target_macs_downsample, target_macs_resnet, target_macs_upsample
            del remain_in_nc
        else:
            cur_macs_front, remain_in_nc = self.netG_student.get_macs_front()
            cur_macs_downsample, remain_in_nc = self.netG_student.get_macs_downsample(remain_in_nc)
            cur_macs_resnet, remain_in_nc = self.netG_student.get_macs_resnet(remain_in_nc)
            cur_macs_upsample, remain_in_nc = self.netG_student.get_macs_upsample(remain_in_nc)

            wandb.log({'cur macs front': cur_macs_front})
            wandb.log({'cur macs downsample': cur_macs_downsample})
            wandb.log({'cur macs resnet': cur_macs_resnet})
            wandb.log({'cur macs upsample': cur_macs_upsample})
            wandb.log({'cur macs': (cur_macs_front + cur_macs_downsample + cur_macs_resnet + cur_macs_upsample)})
            if self.netG_student.pm1.in_channels == 32:
                wandb.log({'cur macs front ratio': cur_macs_front / 0.3083})
                wandb.log({'cur macs downsample ratio': cur_macs_downsample / 0.6040})
                wandb.log({'cur macs resnet ratio': cur_macs_resnet / 1.2929})
                wandb.log({'cur macs upsample ratio': cur_macs_upsample / 2.4159})
                wandb.log({'cur macs ratio': (cur_macs_front + cur_macs_downsample + cur_macs_resnet + cur_macs_upsample) / 4.6211})
            elif self.netG_student.pm1.in_channels == 64:
                wandb.log({'cur macs front ratio': cur_macs_front / 0.6166})
                wandb.log({'cur macs downsample ratio': cur_macs_downsample / 2.4159})
                wandb.log({'cur macs resnet ratio': cur_macs_resnet / 5.0017})
                wandb.log({'cur macs upsample ratio': cur_macs_upsample / 9.6637})
                wandb.log({'cur macs ratio': (cur_macs_front + cur_macs_downsample + cur_macs_resnet + cur_macs_upsample) / 17.6979})
            self.cur_macs = cur_macs_front + cur_macs_downsample + cur_macs_resnet + cur_macs_upsample
            del cur_macs_front, cur_macs_downsample, cur_macs_resnet, cur_macs_upsample
            del remain_in_nc
            self.loss_netG_student_mac_front = torch.tensor([0.]).cuda()
            self.loss_netG_student_mac_downsample = torch.tensor([0.]).cuda()
            self.loss_netG_student_mac_resnet = torch.tensor([0.]).cuda()
            self.loss_netG_student_mac_upsample = torch.tensor([0.]).cuda()
            self.loss_netG_student_mac = torch.tensor([0.]).cuda()
        if not self.opt.no_nuc_loss:
            self.loss_netG_student_nuc = append_loss_nuc(self.netG_student, self.opt.alpha_nuc, self.opt)
        else:
            self.loss_netG_student_nuc = torch.tensor([0.]).cuda()

        self.loss_G = self.loss_G_gan + self.loss_G_recon + self.loss_G_distill + self.loss_netG_student_mac + self.loss_netG_student_nuc
        self.loss_G.backward()

    def optimize_parameters(self, steps):
        raise NotImplementedError

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if hasattr(self, name):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                with open(os.path.join(self.opt.log_dir, name + '.txt'), 'w') as f:
                    f.write(str(net) + '\n')
                    f.write('[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def load_networks(self, verbose=True):
        util.load_network(self.netG_teacher, self.opt.restore_teacher_G_path, verbose)
        if self.opt.restore_student_G_path is not None:
            util.load_network(self.netG_student, self.opt.restore_student_G_path, verbose)
        if self.opt.restore_D_path is not None:
            util.load_network(self.netD, self.opt.restore_D_path, verbose)
        if self.opt.restore_A_path is not None:
            for i, netA in enumerate(self.netAs):
                path = '%s-%d.pth' % (self.opt.restore_A_path, i)
                util.load_network(netA, path, verbose)
        if self.opt.restore_O_path is not None:
            for i, optimizer in enumerate(self.optimizers):
                path = '%s-%d.pth' % (self.opt.restore_O_path, i)
                util.load_optimizer(optimizer, path, verbose)

    def save_networks(self, epoch):

        def save_net(net, save_path):
            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                if isinstance(net, DataParallel):
                    torch.save(net.module.cpu().state_dict(), save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.cpu().state_dict(), save_path)

        save_filename = '%s_net_%s.pth' % (epoch, 'G')
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net%s_student' % 'G')
        save_net(net, save_path)

        save_filename = '%s_net_%s.pth' % (epoch, 'D')
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net%s' % 'D')
        save_net(net, save_path)

        for i, net in enumerate(self.netAs):
            save_filename = '%s_net_%s-%d.pth' % (epoch, 'A', i)
            save_path = os.path.join(self.save_dir, save_filename)
            save_net(net, save_path)

        for i, optimizer in enumerate(self.optimizers):
            save_filename = '%s_optim-%d.pth' % (epoch, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(optimizer.state_dict(), save_path)

    def evaluate_model(self, step):
        raise NotImplementedError

    def test(self):
        with torch.no_grad():
            self.forward()
