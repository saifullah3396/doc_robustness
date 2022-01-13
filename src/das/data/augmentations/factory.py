import dataclasses
from enum import Enum
from typing import List

import numpy as np
import yaml
from das.data.augmentations.augmentations import *
from matplotlib import pyplot as plt


class AugmentationsEnum(str, Enum):
    # basic
    BRIGHTNESS = "brightness"  # 0
    CONTRAST = "contrast"  # 1

    # transforms
    TRANSLATION = "translation"  # 2
    SCALE = "scale"  # 3
    ROTATION = "rotation"  # 4
    AFFINE = "affine"  # 5

    # blurs
    BINARY_BLUR = "binary_blur"  # 6
    GAUSSIAN_BLUR = "gaussian_blur"  # 7
    NOISY_BINARY_BLUR = "noisy_binary_blur"  # 8
    DEFOCUS_BLUR = "defocus_blur"  # 9
    MOTION_BLUR = "motion_blur"  # 10
    ZOOM_BLUR = "zoom_blur"  # 11

    # distortions
    RANDOM_DISTORTION = "random_distortion"  # 12
    RANDOM_BLOTCHES = "random_blotches"  # 13
    SURFACE_DISTORTION = "surface_distortion"  # 14
    THRESHOLD = "threshold"  # 15
    PIXELATE = "pixelate"  # 16
    JPEG_COMPRESSION = "jpeg_compression"  # 17
    ELASTIC = "elastic"  # 18

    # noise
    GAUSSIAN_NOISE_RGB = "gaussian_noise_rgb"  # 19
    SHOT_NOISE_RGB = "shot_noise_rgb"  # 20
    FIBROUS_NOISE = "fibrous_noise"  # 21
    MULTISCALE_NOISE = "multiscale_noise"  # 22


FN_MAP = {
    AugmentationsEnum.BRIGHTNESS: brightness,
    AugmentationsEnum.CONTRAST: contrast,
    AugmentationsEnum.TRANSLATION: translation,
    AugmentationsEnum.SCALE: scale,
    AugmentationsEnum.ROTATION: rotation,
    AugmentationsEnum.AFFINE: affine,
    AugmentationsEnum.BINARY_BLUR: binary_blur,
    AugmentationsEnum.GAUSSIAN_BLUR: gaussian_noise,
    AugmentationsEnum.NOISY_BINARY_BLUR: noisy_binary_blur,
    AugmentationsEnum.DEFOCUS_BLUR: defocus_blur,
    AugmentationsEnum.MOTION_BLUR: motion_blur,
    AugmentationsEnum.ZOOM_BLUR: zoom_blur,
    AugmentationsEnum.RANDOM_BLOTCHES: random_blotches,
    AugmentationsEnum.RANDOM_DISTORTION: random_distortion,
    AugmentationsEnum.SURFACE_DISTORTION: surface_distortion,
    AugmentationsEnum.THRESHOLD: threshold,
    AugmentationsEnum.PIXELATE: pixelate,
    AugmentationsEnum.JPEG_COMPRESSION: jpeg_compression,
    AugmentationsEnum.ELASTIC: elastic,
    AugmentationsEnum.FIBROUS_NOISE: fibrous_noise,
    AugmentationsEnum.GAUSSIAN_NOISE_RGB: gaussian_noise_rgb,
    AugmentationsEnum.SHOT_NOISE_RGB: shot_noise_rgb,
    AugmentationsEnum.MULTISCALE_NOISE: multiscale_noise,
}


@dataclasses.dataclass
class Augmentation:
    name: str
    params: List[dict] = dataclasses.field(default_factory=lambda: [{}, {}, {}, {}, {}])

    def __post_init__(self):
        self.name = AugmentationsEnum(self.name)
        self.aug_fn = FN_MAP.get(self.name, None)
        if self.aug_fn is None:
            raise ValueError(f"Augmentation [{self.name}] is not supported.")

    def __call__(self, image, severity):
        assert image.max() <= 1.0
        if severity > len(self.params):
            return None
        return self.aug_fn(image, **self.params[severity - 1])


@dataclasses.dataclass
class AugmentationArguments:
    output_aug_dir: str
    cls_name = "aug_args"
    n_parallel_jobs: int = 4
    debug: bool = False
    datasets: List[str] = dataclasses.field(default_factory=lambda: ["test"])
    augmentations: List[Augmentation] = dataclasses.field(
        default_factory=lambda: [{}, {}, {}, {}, {}]
    )
