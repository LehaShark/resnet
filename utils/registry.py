from random import random


class Registry(object):
    def __init__(self, name):
        self.name = name
        self._module_dict = dict()

    def get(self, key, kwargs):
        return self._module_dict.get(key, None)(**kwargs)

    def register_module(self, cls):
        self._module_dict[cls.__name__] = cls
        return cls

# if __name__ == '__main__':
#     pass