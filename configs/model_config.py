import os.path

import torch
from easydict import EasyDict
# if __name__ == '__main__':
#     dict = {'criterion': 'BCELoss', }
from configs.dataset_config import DatasetConfig


class ModelConfig:
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


        # self.stage = {
        #
        # }


    def get_input_stem(self, output_channels):
        return [('Conv2d', dict(in_channels=1))]

    # def get_residual_block(self):



class TrainerConfig:
    def __init__(self):
        self.epoch_size = 10
        self.lr = 5e-4
        self.weight_decay = 5e-4
        self.batch_size = 64

        self.show_statistics = True
        self.epoch_num = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = 'CrossEntropyLoss'
        self.optim = 'SGD'
        self.momentum = 0.9
        self.show_each = 1

        self.LOG_PATH = os.path.join(DatasetConfig().PATH, '../logs')