from typing import Type

import torch
from torch import nn

from configs.model_config import OriginalResNetConfig, ModifyResNetConfig
from netlib.blocks import MultipleBlock, InputStem
from utils import StochasticDepth
import torch.nn.functional as F


class ResNet50(nn.Module):
    def __init__(self, config, stochastic_depth: Type[StochasticDepth] = None):
        super().__init__()
        self.config = config

        self._num_blocks = sum(self.config.multipleblock_params.count)
        self._probabilities = []


        if stochastic_depth is not None:
            prob_cls = stochastic_depth(self._num_blocks)
            probs = prob_cls.get_props()
            idx = 0
            for n in self.config.multipleblock_params.count:
                probs[idx] = 1.0
                self._probabilities.append(probs[idx:idx+n])
                idx += n

        # else:
        #     for n in self.config.multipleblock_params.count:
        #         probs[idx] = 1.0
        #         self._probabilities.append(probs[idx:idx + n])
        #         idx += n

        self.input = InputStem(**config.stem)
        # for n in range(num)
        self.stage1 = MultipleBlock(self.input.get_output_channel(),
                                    self.config.multipleblock_params.depth_size[0],
                                    self.config.multipleblock_params.count[0],
                                    self.config,
                                    downsample=self.downsample,
                                    is_first_stage=True,
                                    probabilities=self._probabilities[0])
        for i in range(1, len(config.multipleblock_params.count)):
            in_channels = getattr(self, f'stage{i}').get_output_channel()
            setattr(self, f'stage{i + 1}', MultipleBlock(in_channels,
                                                         self.config.multipleblock_params.depth_size[i],
                                                         self.config.multipleblock_params.count[i],
                                                         self.config,
                                                         downsample=self.downsample,
                                                         probabilities=self._probabilities[i]))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stage4.get_output_channel(), 12)

    def downsample(self, input_channels, output_channels, kernel_size, stride):
        downsample = [nn.Conv2d(input_channels, output_channels, kernel_size, stride=1)]
        if stride == 2:
            downsample.insert(0, nn.AvgPool2d(2, stride))
        return nn.Sequential(*downsample)

    def forward(self, x):
        x = self.input(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.fc(torch.flatten(x, 1))
        return x


if __name__ == '__main__':
    from torchsummary import summary

    cfg = OriginalResNetConfig()
    cfg = ModifyResNetConfig()

    model = ResNet50(cfg)

    summary = summary(model, (3, 224, 224), batch_size=32)
    print(model)
