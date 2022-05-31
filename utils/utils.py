from torch import nn


def get_weights(model):
    _, weights = [], []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            weights.append(param)
        else:
            _.append(param)
    return weights, _

def zero_weights_bn(model):
    for name, module in model.__dict__['_modules'].items():
        if 'stage_' in name:
            last_bn = [[lay for lay in blocks.block_layers if 'BatchNorm' in type(lay).__name__][-1]
                               for blocks in module]

            for batch_norm in last_bn:
                nn.init.zeros_(batch_norm.weight)