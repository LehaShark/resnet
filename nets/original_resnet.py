import torch
from torch import nn

from configs.model_config import ModelConfig
from netlib.blocks import MultipleBlock, InputStem
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input = InputStem(**config.stem)
        # for n in range(num)
        self.stage1 = MultipleBlock(self.input.get_output_channel(), self.config.multipleblock_params.depth_size[0], self.config.multipleblock_params.count[0], self.config, is_first_stage=True)
        self.stage2 = MultipleBlock(self.stage1.get_output_channel(), self.config.multipleblock_params.depth_size[1], self.config.multipleblock_params.count[1], self.config)
        self.stage3 = MultipleBlock(self.stage2.get_output_channel(), self.config.multipleblock_params.depth_size[2], self.config.multipleblock_params.count[2], self.config)
        self.stage4 = MultipleBlock(self.stage3.get_output_channel(), self.config.multipleblock_params.depth_size[3], self.config.multipleblock_params.count[3], self.config)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.stage4.get_output_channel(), 12)


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

    cfg = ModelConfig()
    model = ResNet50(cfg)

    summary = summary(model, (3, 224, 224), batch_size=32)
    print(model)