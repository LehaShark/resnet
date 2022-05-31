import os.path

import torch
from easydict import EasyDict
# if __name__ == '__main__':
#     dict = {'criterion': 'BCELoss', }
from torch import nn

from configs.dataset_config import DatasetConfig


class OriginalResNetConfig:
    def __init__(self):
        # self.class_block_name = 'BaseBlock'
        # self.blocks_count = 4


        self.stem = {'input_channels': 3,
                     'output_channels': 64,
                     'conv_size': (7,),
                     'maxpool_size': 3,
                     'stride': (2, 2)}

        self.baseblock_params = EasyDict()
        self.baseblock_params.conv_size = (1, 3, 1)
        self.baseblock_params.stride = (2, 1, 1)

        self.multipleblock_params = EasyDict()
        self.multipleblock_params.depth_size = [64, 128, 256, 512]
        self.multipleblock_params.count = [3, 4, 6, 3]




    def get_input_stem(self, output_channels):
        return [('Conv2d', dict(in_channels=1))]

class ModifyResNetConfig:
    def __init__(self):

        self.stem = {'input_channels': 3,
                     'output_channels': 64,
                     'conv_size': (3, 3, 3), # mod
                     'maxpool_size': 3,
                     'stride': (2, 1, 1, 2)}

        self.baseblock_params = EasyDict()
        self.baseblock_params.conv_size = (1, 3, 1)
        self.baseblock_params.stride = (1, 2, 1) # mod

        self.multipleblock_params = EasyDict()
        self.multipleblock_params.depth_size = [64, 128, 256, 512]
        self.multipleblock_params.count = [3, 4, 6, 3]


