import torch
from torch import Tensor, nn


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        mean = torch.tensor(self.mean).reshape((1, -1, 1, 1))
        std = torch.tensor(self.std).reshape((1, -1, 1, 1))
        return tensor * std + mean

