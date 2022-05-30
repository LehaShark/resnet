


def get_weights(model):
    _, weights = [], []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            weights.append(param)
        else:
            _.append(param)
    return weights, _