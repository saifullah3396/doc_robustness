"""
Defines the IIT-CDIP dataset.
"""

import os

import pandas as pd
from das.data.datasets.image_dataset_base import ImageDatasetsBase
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)
from das.data.datasets.utils import DataKeysEnum


class IITCDIPDataset(ImageDatasetsBase):
    """IIT-CDIP dataset."""

    is_downloadable = False
    has_val_split = True
    supported_splits = ["train", "test", "val"]

    # define dataset labels
    LABELS = [
        "letter",
        "form",
        "email",
        "handwritten",
        "advertisement",
        "scientific report",
        "scientific publication",
        "specification",
        "file folder",
        "news article",
        "budgetv",
        "invoice",
        "presentation",
        "questionnaire",
        "resume",
        "memo",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_dataset(self):
        if self.split not in self.supported_splits:
            raise ValueError(f"Split argument '{self.split}' not supported.")

        # load the annotations
        data_columns = [DataKeysEnum.IMAGE_FILE_PATH, DataKeysEnum.LABEL]
        data = pd.read_csv(
            self.root_dir / f"labels/{self.split}.txt",
            names=data_columns,
            delim_whitespace=True,
        )
        data[DataKeysEnum.IMAGE_FILE_PATH] = [
            f"{self.root_dir}/images/{x}" for x in data[DataKeysEnum.IMAGE_FILE_PATH]
        ]

        return data
