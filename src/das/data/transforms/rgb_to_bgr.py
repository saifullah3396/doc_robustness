
import torch


class RGBToBGR(torch.nn.Module):
    """
    Applies the transformation on an image to convert grayscale to rgb
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        return sample.permute(2, 1, 0)
