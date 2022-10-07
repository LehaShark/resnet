import torch
from torch import nn
from torch.nn import functional as F

class LabelSmoothing(nn.Module):
    def __init__(self, num_classes: int, smooth: float):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, label):
        if not hasattr(label, '__len__'):
            label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        equal = (1. - self.smooth) * label
        otherwise = (label < 1) * self.smooth / (self.num_classes - 1)
        return equal + otherwise

class ToOneHot(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, target):
        return F.one_hot(target, num_classes=self.classes).transpose(1, -1).squeeze(-1)