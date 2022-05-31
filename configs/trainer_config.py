import os

import torch
from configs.dataset_config import DatasetConfig


class TrainerConfig:
    def __init__(self):
        self.epoch_size = 10
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 64

        self.show_statistics = True
        self.epoch_num = 20
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.criterion = 'CrossEntropyLoss'
        self.optim = 'SGD'
        self.momentum = 0.9
        self.show_each = 10

        self.label_smoothing = 5e-3

        self.LOG_PATH = os.path.join(DatasetConfig().PATH, '../logs')