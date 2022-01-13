import random
import typing
from typing import Optional

import torch
from torchvision.transforms.functional import resize


class RandomRescale(torch.nn.Module):
    """
    Randomly rescales the images based on the max/min dims.

    Args:
        rescale_dims (list): Possible random scale dims for shorter dim
        max_rescale_dim (bool): Maximum rescale dimension for larger dim
        random_sample_max_iters (int): Maximum random sampling iterations
    """

    def __init__(self,
                 # [320, 416, 512, 608, 704]
                 rescale_dims: typing.List[int],
                 max_rescale_dim: int,
                 random_sample_max_iters: Optional[int] = 100):
        super().__init__()

        self.rescale_dims = rescale_dims
        self.max_rescale_dim = max_rescale_dim
        self.random_sample_max_iters = random_sample_max_iters

    def forward(self, sample):
        # randomly rescale the image in the batch as done in ViBertGrid
        # shape (C, H, W)
        image_height = sample.shape[1]
        image_width = sample.shape[2]

        # get larger dim
        larger_dim_idx = 0 if image_height > image_width else 1
        smaller_dim_idx = 0 if image_height < image_width else 1

        rescale_dims = [
            i for i in self.rescale_dims]

        # find random rescale dim
        rescaled_shape = None
        for iter in range(self.random_sample_max_iters):
            if len(rescale_dims) > 0:
                # get smaller dim out of possible dims
                idx, smaller_dim = random.choice(
                    list(enumerate(rescale_dims)))

                # find the rescale ratio
                rescale_ratio = smaller_dim / \
                    sample.shape[smaller_dim_idx]

                # rescale larger dim
                larger_dim = rescale_ratio * \
                    sample.shape[larger_dim_idx]

                # check if larger dim is smaller than max large
                if larger_dim > self.max_rescale_dim:
                    rescale_dims.pop(idx)
                else:
                    rescaled_shape = list(sample.shape)
                    rescaled_shape[larger_dim_idx] = int(larger_dim)
                    rescaled_shape[smaller_dim_idx] = int(smaller_dim)
                    break
            else:
                # if no smaller dim is possible rescale image according to
                # larger dim
                larger_dim = self.max_rescale_dim

                # find the rescale ratio
                rescale_ratio = larger_dim / \
                    sample.shape[larger_dim_idx]

                # rescale smaller dim
                smaller_dim = rescale_ratio * \
                    sample.shape[smaller_dim_idx]

                rescaled_shape = list(sample.shape)
                rescaled_shape[larger_dim_idx] = int(larger_dim)
                rescaled_shape[smaller_dim_idx] = int(smaller_dim)
                break

        if rescaled_shape is not None:
            # resize the image according to the output shape
            return resize(sample, rescaled_shape[1:])
        else:
            return sample
