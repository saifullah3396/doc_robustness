"""
Defines the DataModule for RVLCDIP-D dataset.
"""

from das.data.data_modules.base import BaseDataModule
from das.data.datasets.impl.rvlcdip_d import RVLCDIPDDataset
from das.data.transforms.transforms import ImageTransformsMixin


class RVLCDIPDDataModule(BaseDataModule, ImageTransformsMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get labels for this task
        self.labels = self.dataset_class.LABELS
        if self.labels is not None:
            self.num_labels = len(self.labels)

    @property
    def dataset_class(self):
        return RVLCDIPDDataset


DATA_MODULE = RVLCDIPDDataModule
