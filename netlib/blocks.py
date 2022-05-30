import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod

from configs.model_config import OriginalResNetConfig
from utils import Registry

REGISTRY = Registry('blocks')

def mish(x):
    return (x * torch.tanh(F.softplus(x)))

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_output_channel(self):
        raise NotImplementedError

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                    # kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


@REGISTRY.register_module
class InputStem(ResidualBlock):
    def __init__(self, input_channels: int, output_channels: int, conv_size: tuple, maxpool_size: int, stride: tuple):
        super().__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.conv_size = conv_size
        self.stride = stride
        self.maxpool_size = maxpool_size

        # num = 1
        setattr(self, f'conv1', nn.Conv2d(self.input_channels, self.output_channels, self.conv_size[0], stride=stride[0], padding=3))
        if len(conv_size) > 1:
            for num, size in enumerate(conv_size):
                if num == 0:
                    continue

                setattr(self, f'conv{num + 1}', nn.Conv2d(self.output_channels, self.output_channels, stride=stride[num], padding=1)
                # self.conv1 = nn.Conv2d(self.input_size, self.output_size, *self.conv_size, stride=self.stride[0])

        self.bn = nn.BatchNorm2d(self.output_channels)

        self.pool = nn.MaxPool2d(self.maxpool_size, stride=self.stride[-1], padding=1)

        self._init_params()
    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = self.pool(x)
        return x

    def get_output_channel(self):
        return self.output_channels


@REGISTRY.register_module
class BaseBlock(ResidualBlock):
    def __init__(self, input_channels: int, output_channels: int, conv_size: tuple = None, stride: tuple = None, is_downsample: bool = False):
        super().__init__()
        self.is_downsample = is_downsample
        if 2*input_channels == output_channels:
            self.stride = 2
        else:
            self.stride = 1
        self.downsample = nn.Conv2d(input_channels, output_channels, 1, stride=self.stride)

        output_channels = output_channels // 4


        # self.downsize = nn.Conv2d(input_channels, output_channels, 1, stride=s)


        self.convIn = nn.Conv2d(input_channels, output_channels, conv_size[0], stride=stride[0])
        self.convHid = nn.Conv2d(output_channels, output_channels, conv_size[1], stride=stride[1], padding=1)
        self.convOut = nn.Conv2d(output_channels, 4 * output_channels, conv_size[2], stride=stride[2])

        self.bn_in = nn.BatchNorm2d(output_channels)
        self.bn_out = nn.BatchNorm2d(4 * output_channels)

        self._init_params()

    def forward(self, x):
        z = F.relu(self.bn_in(self.convIn(x)))
        z = F.relu(self.bn_in(self.convHid(z)))
        z = self.bn_out(self.convOut(z))
        if self.is_downsample:
            return F.relu(self.downsample(x) + z)
        return F.relu(x + z)


class MultipleBlock(ResidualBlock):
    def __init__(self, input_channels: int, output_channels: int, blocks_count: int, config, is_first_stage: bool = False):
        super().__init__()
        self.config = config
        self.input_channels = input_channels
        self.output_channels = 4 * output_channels

        self.stride = (1, 1, 1)


        self.block_init = {'input_channels': self.input_channels,
                           'output_channels': self.output_channels,
                           'conv_size': self.config.baseblock_params.conv_size,
                           'stride': self.stride if is_first_stage else self.config.baseblock_params.stride,
                           'is_downsample': True}

        setattr(self, 'block1', REGISTRY.get('BaseBlock', self.block_init))


        self.block_init['is_downsample'] = False
        self.block_init['input_channels'] = self.output_channels
        self.block_init['stride'] = self.stride


        for num in range(2, blocks_count + 1):
            setattr(self, f'block{num}', REGISTRY.get('BaseBlock', self.block_init))


        super(MultipleBlock, self)._init_params()

    def get_output_channel(self):
        return self.output_channels

    def forward(self, x):
        for name, module in self.__dict__['_modules'].items():
            # todo: enumerator stage start
            if 'block' in name:
                x = module(x)
        return x


if __name__ == '__main__':
    config = OriginalResNetConfig()
    m = MultipleBlock(64, 4, config)
    x = torch.from_numpy(np.random.normal(size=(1, 64, 56, 56)))
    m.forward(x)
