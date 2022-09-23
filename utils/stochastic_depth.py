from abc import abstractmethod

import numpy as np


class StochasticDepth:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers

    def __call__(self, *args, **kwargs):
        self.get_props(*args, **kwargs)

    @abstractmethod
    def get_props(self, *args, **kwargs):
        raise NotImplementedError()


class LinearStochasticDepth(StochasticDepth):
    def __init__(self, num_layers: int, p_min: float = 0.5):
        super().__init__(num_layers)
        self.p_min = p_min

    def get_props(self):
        probs = 1 - np.linspace(0, 1, self.num_layers) * (1 - self.p_min)
        return probs.tolist()
