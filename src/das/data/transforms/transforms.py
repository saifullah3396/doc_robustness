"""
Defines different types of data transformations.
"""


from typing import Callable

import torch
from das.data.transforms.dict_transform import DictTransform
from das.data.transforms.grayscale_to_rgb import GrayScaleToRGB
from das.data.transforms.rgb_to_bgr import RGBToBGR
from das.utils.basic_utils import create_logger
from torchvision import transforms
from torchvision.transforms.transforms import ConvertImageDtype, Normalize

# setup logging
logger = create_logger(__name__)


class TransformsMixin:
    def default_transforms(self):
        """ Default transform for the dataset """
        return {
            'train': None,
            'val': None,
            'test': None
        }


class ImageTransformsMixin(TransformsMixin):
    """
    Defines the default transformations to be used for images. Can be modified based
    on config.
    """

    def default_transforms(self) -> Callable:
        data_transforms_args = self.data_args.data_transforms_args
        if data_transforms_args is None:
            return {
                'train': None,
                'val': None,
                'test': None
            }

        def_t = {
            'train': [],
            'test': [],
            'val': []
        }
        for k, v in def_t.items():
            # convert to 3 channels if grayscale image
            if data_transforms_args.convert_grayscale_to_rgb:
                v.append(DictTransform(['image'], GrayScaleToRGB()))

            # convert rgb to bgr
            if data_transforms_args.convert_rgb_to_bgr:
                v.append(DictTransform(['image'], RGBToBGR()))

            # rescale images
            if k == 'train':
                if data_transforms_args.train_image_rescale_strategy is not None:
                    v.append(
                        DictTransform(
                            ['image'],
                            data_transforms_args.train_image_rescale_strategy.create()))
            else:
                if data_transforms_args.eval_image_rescale_strategy is not None:
                    v.append(
                        DictTransform(
                            ['image'],
                            data_transforms_args.eval_image_rescale_strategy.create()))

            # convert type
            v.append(
                DictTransform(
                    ['image'],
                    ConvertImageDtype(torch.float)))

            # normalize image
            if data_transforms_args.normalize_dataset:
                if data_transforms_args.dataset_mean is None or \
                        data_transforms_args.dataset_std is None:
                    raise ValueError(
                        'dataset_mean and dataset_std must be defined for normalization of '
                        'the dataset')

                v.append(
                    DictTransform(
                        ['image'],
                        Normalize(
                            data_transforms_args.dataset_mean['image'],
                            data_transforms_args.dataset_std['image'])))

        for k, v in def_t.items():
            def_t[k] = torch.nn.Sequential(*transforms.Compose(v).transforms)

        return def_t
