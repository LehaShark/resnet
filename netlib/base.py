# from torch import nn
# import torch.nn.functional as F
#
# class BasicBlock(nn.Module):
#     def __init__(self, input_channels: int, output_channels: int, stride: int = None):
#         output_channels = output_channels // 4
#         self.conv1 = nn.Conv2d(input_channels, output_channels, 1, stride=2)
#         self.conv2 = nn.Conv2d(output_channels, output_channels, 3)
#         self.conv3 = nn.Conv2d(output_channels, 4 * output_channels, 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         return x
#
# class BaseMultipleBlock(nn.Module):
#     def __init__(self, bloks_count: int, input_channels: int):
#         self.input_channels = input_channels
#         setattr(self, 'block1', BasicBlock(self.input_channels, 4 * self.input_channels))
#         for num in range(2, bloks_count + 1):
#             setattr(self, f'block{num}', BasicBlock(4 * self.input_channels, self.input_channels))
#             # self.mult1 = BasicBlock()
#             # self.mult2 = BasicBlock()
#             # self.mult3 = BasicBlock()