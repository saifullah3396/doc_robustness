"""
Defines the Tobacco3842-D dataset.
"""

import os
from pathlib import Path

import pandas as pd
import tqdm
from das.data.augmentations.factory import AugmentationsEnum
from das.data.datasets.image_dataset_base import ImageDatasetsBase
from das.data.datasets.impl.tobacco3842 import Tobacco3842Dataset
from das.data.datasets.utils import DataKeysEnum
from das.utils.basic_utils import create_logger

logger = create_logger(__name__)


class Tobacco3482DDataset(Tobacco3842Dataset):
    """Tobacco3482-D dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_dataset(self):
        test_set_names = pd.read_csv(self.root_dir / "test_set.txt").values.tolist()
        test_set_names = [elem for l in test_set_names for elem in l]

        data = []
        labels_root_dir = self.root_dir / "images" / "Tobacco3482"
        for dir in os.listdir(labels_root_dir):
            if dir in self.LABELS:
                for file in os.listdir(labels_root_dir / dir):
                    if file in test_set_names:
                        sample = []

                        file_path = Path(labels_root_dir / dir / file)

                        # add image path
                        sample.append(str(file_path))

                        # add label
                        label_idx = self.LABELS.index(dir)
                        sample.append(label_idx)

                        # add sample to data
                        data.append(sample)

        # convert data list to df
        data_columns = [DataKeysEnum.IMAGE_FILE_PATH, DataKeysEnum.LABEL]
        data = pd.DataFrame(data, columns=data_columns)

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
