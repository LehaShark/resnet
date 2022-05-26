import os
import time

import torch


class Config(object):
    def __init__(self, batch_size, lr, dataset_name, model_name, num_classes, dataset_path: str = None,
                 debug: bool = True, show_each: int = 1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 seed=None,
                 overfit=False):

        self.dataset_path = dataset_path
        self.debug = debug
        self.batch_size = batch_size
        self.lr = lr
        self.show_each = show_each
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.overfit = overfit
        self.device = device
        self.num_classes = num_classes

        self.seed = seed

        experiment_name = f'model_{self.model_name}_batch_size{self.batch_size}_lr_{self.lr}_{time.time()}'
        self.LOG_PATH = os.path.join('../logs', self.dataset_name, experiment_name)

