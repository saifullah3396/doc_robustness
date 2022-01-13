"""
Defines the RVLCDIP-D dataset.
"""

import os
from pathlib import Path

import pandas as pd
import tqdm
from das.data.augmentations.factory import AugmentationsEnum
from das.data.datasets.impl.rvlcdip import RVLCDIPDataset
from das.data.datasets.utils import DataKeysEnum
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


class RVLCDIPDDataset(RVLCDIPDataset):
    """RVLCDIP-D distortion dataset based on https://www.cs.cmu.edu/~aharley/rvl-cdip/."""

    supported_splits = ["test"]

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
            f"{self.root_dir}/images/{x[:-4]}/"
            for x in data[DataKeysEnum.IMAGE_FILE_PATH]
        ]

        # now generate the augmented data from the set
        augmented_data = []
        augmentations_list = list(AugmentationsEnum)
        for idx, row in tqdm.tqdm(data.iterrows()):
            image_base_path = row[DataKeysEnum.IMAGE_FILE_PATH]
            for augmentation in os.listdir(image_base_path):
                if augmentation.endswith(".jpg"):
                    sample = []
                    file_path = str(Path(image_base_path) / augmentation)
                    sample.append(file_path)
                    sample.append(row[DataKeysEnum.LABEL])
                    sample.append(-1)
                    sample.append(-1)
                    augmented_data.append(sample)
                    continue

                for severity in os.listdir(Path(image_base_path) / augmentation):
                    sample = []
                    file_path = str(Path(image_base_path) / augmentation / severity)
                    sample.append(file_path)
                    sample.append(row[DataKeysEnum.LABEL])
                    sample.append(
                        augmentations_list.index(AugmentationsEnum(augmentation))
                    )
                    sample.append(int(severity[:-4]))
                    augmented_data.append(sample)

        data_columns = [
            DataKeysEnum.IMAGE_FILE_PATH,
            DataKeysEnum.LABEL,
            DataKeysEnum.AUGMENTATION,
            DataKeysEnum.SEVERITY,
        ]
        return pd.DataFrame(augmented_data, columns=data_columns)
