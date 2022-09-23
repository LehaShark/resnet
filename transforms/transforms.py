import torch
from torch import nn
from torch.nn import functional as F

class LabelSmoothing(nn.Module):
    def __init__(self, num_classes: int, smooth: float):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, target):
        if not hasattr(target, '__len__'):
            target = F.one_hot(torch.tensor(target), num_classes=self.num_classes)
        equal = (1. - self.smooth) * target
        otherwise = (target < 1) * self.smooth / (self.num_classes - 1)
        return equal + otherwise