import random
import typing
from typing import Optional

import torch
from torchvision.transforms.functional import resize


class Rescale(torch.nn.Module):
    """
    Randomly rescales the images based on the max/min dims.

    Args:
        rescale_dim (int): Rescale dimension for smaller dim
        rescale_smaller_dim (bool): Whether to rescale smaller dim, otherwise larger
            dimension is scaled
    """

    def __init__(
            self,
            rescale_dim: int,
            rescale_smaller_dim: bool = True,
            rescale_both_dims: bool = False):
        super().__init__()

        self.rescale_dim = rescale_dim
        self.rescale_smaller_dim = rescale_smaller_dim
        self.rescale_both_dims = rescale_both_dims

    def forward(self, sample):
        # randomly rescale the image in the batch as done in ViBertGrid
        # shape (C, H, W)
        image_height = sample.shape[1]
        image_width = sample.shape[2]

        if not self.rescale_both_dims:
            # get smaller dim
            larger_dim_idx = 0 if image_height > image_width else 1
            smaller_dim_idx = 0 if image_height < image_width else 1

            dim_idx = smaller_dim_idx if self.rescale_smaller_dim else larger_dim_idx
            other_dim_idx = \
                larger_dim_idx if self.rescale_smaller_dim else smaller_dim_idx

            # find the rescale ratio
            rescale_ratio = self.rescale_dim / sample.shape[dim_idx]

            # rescale the other dim
            other_dim = rescale_ratio * sample.shape[other_dim_idx]

            rescaled_shape = list(sample.shape)
            rescaled_shape[dim_idx] = int(self.rescale_dim)
            rescaled_shape[other_dim_idx] = int(other_dim)
        else:
            rescaled_shape = list(sample.shape)
            rescaled_shape[1] = self.rescale_dim
            rescaled_shape[2] = self.rescale_dim

        # resize the image according to the output shape
        return resize(sample, rescaled_shape[1:])
