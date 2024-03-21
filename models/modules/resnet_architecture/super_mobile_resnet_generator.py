import functools

from torch import nn

from models.modules.super_modules import SuperConvTranspose2d, SuperConv2d, SuperSeparableConv2d
from models.networks import BaseNetwork
import torch.nn.functional as F
import torch

class SuperMobileResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        super(SuperMobileResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout_rate, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SuperSeparableConv2d(in_channels=dim, out_channels=dim,
                                 kernel_size=3, padding=p, stride=1),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SuperSeparableConv2d(in_channels=dim, out_channels=dim,
                                 kernel_size=3, padding=p, stride=1),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, input, config):
        x = input
        cnt = 0
        for module in self.conv_block:
            if isinstance(module, SuperSeparableConv2d):
                if cnt == 1:
                    config['channel'] = input.size(1)
                x = module(x, config)
                cnt += 1
            else:
                x = module(x)
        out = input + x
        return out


class SuperMobileResnetBlock_with_SPM(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, dropout_rate, use_bias, SPM):
        super(SuperMobileResnetBlock_with_SPM, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, dropout_rate, use_bias)
        self.spm = SPM  # SPM
        self.pm = BinaryConv2d(in_channels=dim, out_channels=dim, groups=dim)  # PM
    def build_conv_block(self, dim, padding_type, norm_layer, dropout_rate, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SuperSeparableConv2d(in_channels=dim, out_channels=dim,
                                 kernel_size=3, padding=p, stride=1),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        conv_block += [nn.Dropout(dropout_rate)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [
            SuperSeparableConv2d(in_channels=dim, out_channels=dim,
                                 kernel_size=3, padding=p, stride=1),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, input, config):
        x = input
        cnt = 0
        for module in self.conv_block:
            if isinstance(module, SuperSeparableConv2d):
                if cnt == 1:
                    config['channel'] = input.size(1)
                x = module(x, config)
                cnt += 1
            else:
                x = module(x)
                if isinstance(module, nn.ReLU):  # pm after ReLU
                    x = self.pm(x)
        out = input + x
        out = self.spm(out)  # SPM
        return out

    def get_macs(self, remain_in_nc):
        # get first SuperSeperableConv2d macs
        # get pm remain channels
        w = self.pm.weight.detach()
        binary_w = (w > 0.5).float()
        residual = w - binary_w
        branch_out = self.pm.weight - residual
        remain_out_nc_pm = torch.sum(torch.squeeze(branch_out))
        first_macs = self.conv_block[1].get_macs(remain_in_nc, remain_out_nc_pm)
        # get second SuperSeprableConv2d macs
        # get spm remain channels
        w = self.spm.weight.detach()
        binary_w = (w > 0.5).float()
        residual = w - binary_w
        branch_out = self.pm.weight - residual
        remain_out_nc_spm = torch.sum(torch.squeeze(branch_out))
        second_macs = self.conv_block[6].get_macs(remain_out_nc_pm, remain_out_nc_spm)
        return first_macs + second_macs, remain_out_nc_spm


class SuperMobileResnetGenerator(BaseNetwork):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.BatchNorm2d, dropout_rate=0, n_blocks=6,
                 padding_type='reflect'):
        assert n_blocks >= 0
        super(SuperMobileResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 SuperConv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [SuperConv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        for i in range(n_blocks1):
            model += [SuperMobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias)]

        for i in range(n_blocks2):
            model += [SuperMobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias)]

        for i in range(n_blocks3):
            model += [SuperMobileResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [SuperConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1,
                                           bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [SuperConv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        configs = self.configs
        input = input.clamp(-1, 1)
        x = input
        cnt = 0
        for i in range(0, 10):
            module = self.model[i]
            if isinstance(module, SuperConv2d):
                channel = configs['channels'][cnt] * (2 ** cnt)
                config = {'channel': channel}
                x = module(x, config)
                cnt += 1
            else:
                x = module(x)
        for i in range(3):
            for j in range(10 + i * 3, 13 + i * 3):
                if len(configs['channels']) == 6:
                    channel = configs['channels'][3] * 4
                else:
                    channel = configs['channels'][i + 3] * 4
                config = {'channel': channel}
                module = self.model[j]
                x = module(x, config)
        cnt = 2
        for i in range(19, 28):
            module = self.model[i]
            if isinstance(module, SuperConvTranspose2d):
                cnt -= 1
                if len(configs['channels']) == 6:
                    channel = configs['channels'][5 - cnt] * (2 ** cnt)
                else:
                    channel = configs['channels'][7 - cnt] * (2 ** cnt)
                config = {'channel': channel}
                x = module(x, config)
            elif isinstance(module, SuperConv2d):
                config = {'channel': module.out_channels}
                x = module(x, config)
            else:
                x = module(x)
        return x


class BinaryConv2d(nn.Conv2d):
    """docstring for QuanConv"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           groups, bias)
        nn.init.constant_(self.weight, 0.6)

    def forward(self, x):
        # weight = self.weight
        w = self.weight.detach()
        binary_w = (w > 0.5).float()
        residual = w - binary_w
        weight = self.weight - residual

        output = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output


class SuperMobileResnetGenerator_with_SPM(BaseNetwork):
    def __init__(self, input_nc, output_nc, ngf, norm_layer=nn.BatchNorm2d, dropout_rate=0, n_blocks=6,
                 padding_type='reflect'):
        assert n_blocks >= 0
        super(SuperMobileResnetGenerator_with_SPM, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 SuperConv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [SuperConv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling

        n_blocks1 = n_blocks // 3
        n_blocks2 = n_blocks1
        n_blocks3 = n_blocks - n_blocks1 - n_blocks2

        self.spm1 = BinaryConv2d(in_channels=ngf * mult, out_channels=ngf * mult, groups=ngf * mult)
        self.spm2 = BinaryConv2d(in_channels=ngf * mult, out_channels=ngf * mult, groups=ngf * mult)
        self.spm3 = BinaryConv2d(in_channels=ngf * mult, out_channels=ngf * mult, groups=ngf * mult)

        for i in range(n_blocks1):
            model += [SuperMobileResnetBlock_with_SPM(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias, SPM=self.spm1)]

        for i in range(n_blocks2):
            model += [SuperMobileResnetBlock_with_SPM(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias, SPM=self.spm2)]

        for i in range(n_blocks3):
            model += [SuperMobileResnetBlock_with_SPM(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                             dropout_rate=dropout_rate,
                                             use_bias=use_bias, SPM=self.spm3)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [SuperConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1,
                                           bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [SuperConv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        configs = self.configs
        input = input.clamp(-1, 1)
        x = input
        cnt = 0
        for i in range(0, 10):
            module = self.model[i]
            if isinstance(module, SuperConv2d):
                channel = configs['channels'][cnt] * (2 ** cnt)
                config = {'channel': channel}
                x = module(x, config)
                cnt += 1
            else:
                x = module(x)
        for i in range(3):
            for j in range(10 + i * 3, 13 + i * 3):
                if len(configs['channels']) == 6:
                    channel = configs['channels'][3] * 4
                else:
                    channel = configs['channels'][i + 3] * 4
                config = {'channel': channel}
                module = self.model[j]
                x = module(x, config)
        cnt = 2
        for i in range(19, 28):
            module = self.model[i]
            if isinstance(module, SuperConvTranspose2d):
                cnt -= 1
                if len(configs['channels']) == 6:
                    channel = configs['channels'][5 - cnt] * (2 ** cnt)
                else:
                    channel = configs['channels'][7 - cnt] * (2 ** cnt)
                config = {'channel': channel}
                x = module(x, config)
            elif isinstance(module, SuperConv2d):
                config = {'channel': module.out_channels}
                x = module(x, config)
            else:
                x = module(x)
        return x

    def get_macs(self):
        total_macs = torch.tensor([0.0]).cuda()
        remain_in_nc = torch.tensor([128]).cuda()
        for name, module in self.model.named_children():
            if isinstance(module, SuperMobileResnetBlock_with_SPM):
                macs, remain_in_nc = module.get_macs(remain_in_nc)
                total_macs += macs
        return total_macs / 1e9
