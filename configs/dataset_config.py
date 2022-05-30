import os

class DatasetConfig():
    def __init__(self):
        self.PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.train = os.path.join(self.PATH, 'train')
        self.valid = os.path.join(self.PATH, 'valid')