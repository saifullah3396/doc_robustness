"""
Defines the DataModule for Tobacco3842 dataset.
"""

from das.data.data_modules.base import BaseDataModule
from das.data.datasets.impl.tobacco3482c import Tobacco3482CDataset
from das.data.transforms.transforms import ImageTransformsMixin


class Tobacco3482CDataModule(BaseDataModule, ImageTransformsMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get labels for this task
        self.labels = self.dataset_class.LABELS
        if self.labels is not None:
            self.num_labels = len(self.labels)

    @property
    def dataset_class(self):
        return Tobacco3482CDataset


DATA_MODULE = Tobacco3482CDataModule
