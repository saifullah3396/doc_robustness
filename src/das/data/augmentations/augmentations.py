import ctypes
import numbers
import warnings
from io import BytesIO
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import ocrodeg
import scipy.ndimage as ndi
import torch
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
from torchvision.transforms import RandomAffine
from torchvision.transforms.transforms import _check_sequence_input, _setup_angle


class DeterministicAffine(RandomAffine):
    def __init__(
        self,
        degrees,
        translate=None,
        scale=None,
        shear=None,
        interpolation=F.InterpolationMode.NEAREST,
        fill=0,
        fillcolor=None,
        resample=None,
    ):
        torch.nn.Module.__init__(self)

        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = F._interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = F._interpolation_modes_from_int(interpolation)

        if fillcolor is not None:
            warnings.warn(
                "Argument fillcolor is deprecated and will be removed since v0.10.0. Please, use fill instead"
            )
            fill = fillcolor

        self.degrees = degrees

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            if scale <= 0:
                raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2,))
        else:
            self.shear = shear

        self.resample = self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fillcolor = self.fill = fill

    @staticmethod
    def get_params(
        degrees: List[float],
        translate: Optional[List[float]],
        scale: Optional[List[float]],
        shears: Optional[List[float]],
        img_size: List[int],
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = degrees
        if translate is not None:
            tx = float(translate[0] * img_size[0])
            ty = float(translate[1] * img_size[1])
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale is None:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = shears[0]
            shear_y = shears[1]

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def brightness(image, beta):
    return np.clip(image + beta, 0, 1)


def contrast(image, alpha):
    channel_means = np.mean(image, axis=(0, 1))
    return np.clip((image - channel_means) * alpha + channel_means, 0, 1)


def translation(image, magnitude):
    return ocrodeg.transform_image(image, translation=magnitude)


def scale(image, scale, fill=1):
    image = torch.tensor(image).unsqueeze(0)
    scale = np.random.choice(scale)
    scale = [scale - 0.025, scale + 0.025]
    t = RandomAffine(degrees=0, scale=scale, fill=fill)
    image = t(image).squeeze().numpy()
    return image
    # image = torch.tensor(image).unsqueeze(0)
    # t = DeterministicAffine(
    #     degrees=0, translate=(0, 0), shear=(0, 0), scale=scale, fill=fill)
    # image = t(image).squeeze().numpy()
    # return image


def rotation(image, magnitude):
    return ndi.rotate(image, magnitude)


def affine(image, degrees, translate=[0, 0], shear=[0, 0], fill=1):
    image = torch.tensor(image).unsqueeze(0)

    translate = np.random.choice(translate)
    translate = [translate - 0.01, translate + 0.01]

    degrees = np.random.choice(degrees)
    degrees = [degrees - 1, degrees + 1]

    shear = np.random.choice(shear)
    shear = [shear - 0.5, shear + 0.05]

    t = RandomAffine(degrees=degrees, translate=translate, shear=shear, fill=fill)
    image = t(image).squeeze().numpy()
    return image


def binary_blur(image, sigma):
    return ocrodeg.binary_blur(image, sigma=sigma)


def noisy_binary_blur(image, sigma, noise):
    return ocrodeg.binary_blur(image, sigma=sigma, noise=noise)


def defocus_blur(image, radius, alias_blur):
    kernel = disk(radius=radius, alias_blur=alias_blur)
    return np.clip(cv2.filter2D(image, -1, kernel), 0, 1)


def motion_blur(image, size):
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size, dtype=np.float32)
    kernel_motion_blur = cv2.warpAffine(
        kernel_motion_blur,
        cv2.getRotationMatrix2D(
            (size / 2 - 0.5, size / 2 - 0.5), np.random.uniform(-45, 45), 1.0
        ),
        (size, size),
    )
    kernel_motion_blur = kernel_motion_blur * (1.0 / np.sum(kernel_motion_blur))
    return cv2.filter2D(image, -1, kernel_motion_blur)


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(
        img[top : top + ch, left : left + cw], (zoom_factor, zoom_factor), order=1
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top : trim_top + h, trim_left : trim_left + w]


def zoom_blur(image, zoom_factor_start, zoom_factor_end, zoom_factor_step):
    out = np.zeros_like(image)
    zoom_factor_range = np.arange(zoom_factor_start, zoom_factor_end, zoom_factor_step)
    for zoom_factor in zoom_factor_range:
        out += clipped_zoom(image, zoom_factor)
    return np.clip((image + out) / (len(zoom_factor_range) + 1), 0, 1)


def random_distortion(image, sigma, maxdelta):
    noise = ocrodeg.bounded_gaussian_noise(image.shape, sigma, maxdelta)
    return ocrodeg.distort_with_noise(image, noise)


def random_blotches(image, fgblobs, bgblobs, fgscale, bgscale):
    return ocrodeg.random_blotches(
        image, fgblobs=fgblobs, bgblobs=bgblobs, fgscale=fgscale, bgscale=bgscale
    )


def surface_distortion(image, magnitude):
    noise = ocrodeg.noise_distort1d(image.shape, magnitude=magnitude)
    return ocrodeg.distort_with_noise(image, noise)


def threshold(image, magnitude):
    blurred = ndi.gaussian_filter(image, magnitude)
    return 1.0 * (blurred > 0.5)


def gaussian_noise(image, magnitude):
    return ndi.gaussian_filter(image, magnitude)


def gaussian_noise_rgb(image, magnitude):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return np.clip(image + np.random.normal(size=image.shape, scale=magnitude), 0, 1)


def shot_noise_rgb(image, magnitude):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return np.clip(np.random.poisson(image * magnitude) / float(magnitude), 0, 1)


def fibrous_noise(image, blur, blotches):
    return ocrodeg.printlike_fibrous(image, blur=blur, blotches=blotches)


def multiscale_noise(image, blur, blotches):
    return ocrodeg.printlike_multiscale(image, blur=blur, blotches=blotches)


def pixelate(image, magnitude):
    h, w = image.shape
    image = cv2.resize(
        image, (int(w * magnitude), int(h * magnitude)), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)


def jpeg_compression(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode(".jpg", image * 255, encode_param)
    decimg = cv2.imdecode(encimg, 0) / 255.0
    return decimg


def elastic(image, alpha, sigma, alpha_affine, random_state=None):
    assert len(image.shape) == 2
    shape = image.shape
    shape_size = shape[:2]

    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + np.random.uniform(
        -alpha_affine, alpha_affine, size=pts1.shape
    ).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), sigma, mode="reflect", truncate=3
        )
        * alpha
    ).astype(np.float32)
    dy = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), sigma, mode="reflect", truncate=3
        )
        * alpha
    ).astype(np.float32)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return (
        np.clip(
            map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
            0,
            1,
        )
        * 255
    )
