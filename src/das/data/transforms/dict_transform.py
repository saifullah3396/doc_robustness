

import torch


class DictTransform(torch.nn.Module):
    """
    Applies the transformation on given keys for dictionary outputs

    Args:
        keys (list): List of keys
        transform (callable): Transformation to be applied
    """

    def __init__(self, keys: list, transform: callable):
        super().__init__()

        self.keys = keys
        self.transform = transform

    def forward(self, sample):
        for key in self.keys:
            if key in sample:
                sample[key] = self.transform(sample[key])

        return sample
