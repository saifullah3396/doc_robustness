"""
Defines the SROIE dataset.
"""


import json
import os

import pandas as pd
from das.data.data_args import DataArguments
from das.data.datasets.image_dataset_base import ImageDatasetsBase
from das.data.tokenizer import DataTokenizer
from das.utils.basic_utils import create_logger
from PIL import Image
from pycocotools.coco import COCO

logger = create_logger(__name__)


def sort_annotations(anns, image_height):
    num_anns = len(anns)
    bbox_height_mean = 0
    for ann in anns:
        bbox_height_mean += abs(ann["bbox"][3])
    bbox_height_mean /= num_anns

    anns = sorted(anns, key=lambda ann: (ann["bbox"][1], ann["bbox"][0]))

    rows = [[]]
    rows[-1].append(anns[0])

    threshold_value_y = bbox_height_mean / image_height / 2.0
    for i in range(1, num_anns):
        cy1 = (anns[i]["bbox"][1] + anns[i]["bbox"][3] / 2.0) / image_height
        cy0 = (anns[i - 1]["bbox"][1] + anns[i - 1]["bbox"][3] / 2.0) / image_height
        if abs(cy1 - cy0) < threshold_value_y:
            rows[-1].append(anns[i])
        else:
            rows.append([])
            rows[-1].append(anns[i])

    sorted_rows = []
    for row in rows:
        sorted_rows.append(sorted(row, key=lambda ann: (ann["bbox"][0])))

    return [ann for row in sorted_rows for ann in row]


class SroieDataset(ImageDatasetsBase):
    """SROIE dataset from https://rrc.cvc.uab.es/?ch=13/."""

    read_from_coco = True
    is_downloadable = False
    scheme = "IOBES"

    NER_TAGS_S = [
        # other
        "O",
        # company tags
        "S-COMPANY",
        # date tags
        "S-DATE",
        # address tags
        "S-ADDRESS",
        # total tags
        "S-TOTAL",
    ]

    NER_TAGS_IOB2 = [
        # other
        "O",
        # company tags
        "B-COMPANY",
        "I-COMPANY",
        # date tags
        "B-DATE",
        "I-DATE",
        # address tags
        "B-ADDRESS",
        "I-ADDRESS",
        # total tags
        "B-TOTAL",
        "I-TOTAL",
    ]

    # define dataset labels
    NER_TAGS_IOBES = [
        # other
        "O",
        # company tags
        "S-COMPANY",
        "B-COMPANY",
        "I-COMPANY",
        "E-COMPANY",
        # date tags
        "S-DATE",
        "B-DATE",
        "E-DATE",
        "I-DATE",
        # address tags
        "S-ADDRESS",
        "B-ADDRESS",
        "I-ADDRESS",
        "E-ADDRESS",
        # total tags
        "S-TOTAL",
        "B-TOTAL",
        "I-TOTAL",
        "E-TOTAL",
    ]

    @staticmethod
    def get_ner_tags():
        if SroieDataset.scheme == "S":
            return SroieDataset.NER_TAGS_S
        elif SroieDataset.scheme == "IOB2":
            return SroieDataset.NER_TAGS_IOB2
        elif SroieDataset.scheme == "IOBES":
            return SroieDataset.NER_TAGS_IOBES
        else:
            raise ValueError("NER scheme not supported!")

    def __init__(
        self,
        data_args: DataArguments,
        split: str,
        transforms=None,
        download: bool = False,
        use_cached: bool = False,
    ):
        """
        Args:
            root_dir (string): Directory with all the data images and annotations.
            split (string): 'train' or 'test' splits for the data.
            transforms (callable, optional): Optional transform to be applied on a
                sample.
            download (bool): Whether to download the dataset if it does not exist.
            use_cached (bool): Whether to use cached data or prepare it again
        """
        super().__init__(
            data_args,
            split,
            transforms=transforms,
            download=download,
            use_cached=use_cached,
        )

    def _load_dataset(self):
        # get split dataset dir
        if self.split == "train":
            data_dir = self.root_dir / "train_dataset"
        elif self.split == "test":
            # hack that needs to be fixed later on
            data_dir = self.root_dir / "train_dataset"
        else:
            raise ValueError(f"Split argument '{self.split}' not supported.")

        if not data_dir.exists():
            raise ValueError(
                f"Could not find {str(data_dir)} directory in the dataset "
                f"directory: {self.root_dir}"
            )

        if SroieDataset.read_from_coco:
            # get the length of dataset by getting number of files in images dir
            img_dir = data_dir / "images"
            ann_file_path = str(data_dir / "coco.json")
            coco = COCO(annotation_file=ann_file_path)

            data = []
            ner_tags = SroieDataset.get_ner_tags()
            ner_tags_to_idx = dict(zip(ner_tags, range(len(ner_tags))))
            map_cat_id_to_dataset_id = {}
            for cat in coco.dataset["categories"]:
                cat_name = cat["name"]
                if SroieDataset.scheme == "S":
                    cat_name = cat_name.replace("B-", "S-")
                    cat_name = cat_name.replace("E-", "S-")
                    cat_name = cat_name.replace("I-", "S-")
                elif SroieDataset.scheme == "IOB2":
                    cat_name = cat_name.replace("S-", "B-")
                    cat_name = cat_name.replace("E-", "I-")
                map_cat_id_to_dataset_id[cat["id"]] = ner_tags_to_idx[cat_name]

            for im in coco.dataset["images"]:
                file_name = im["file_name"]
                width = im["width"]
                height = im["height"]

                tokens = []
                ner_tags = []
                bboxes = []
                anns = coco.loadAnns(coco.getAnnIds(imgIds=[im["id"]]))
                anns = sort_annotations(anns, height)
                for ann in anns:
                    if "text" not in ann["metadata"]:
                        continue
                    tokens.append(ann["metadata"]["text"])
                    ner_tags.append(map_cat_id_to_dataset_id[ann["category_id"]])
                    x1 = ann["bbox"][0] / width
                    x2 = (ann["bbox"][0] + ann["bbox"][2]) / width
                    y1 = ann["bbox"][1] / height
                    y2 = (ann["bbox"][1] + ann["bbox"][3]) / height
                    bboxes.append([x1, y1, x2, y2])

                image_path = img_dir / file_name
                image_path = str(image_path).replace("json", "jpg")
                data.append([image_path, tokens, bboxes, ner_tags])
        else:
            # get the length of dataset by getting number of files in images dir
            ann_dir = data_dir / "annotations"
            img_dir = data_dir / "images"

            # load the annotations data from json files for all images
            data = []
            ner_tags = SroieDataset.get_ner_tags()
            ner_tags_to_idx = dict(zip(ner_tags, range(len(ner_tags))))

            for file in sorted(os.listdir(ann_dir)):
                # load associated annotations
                tokens = []
                bboxes = []
                ner_tags = []

                ann_file_path = os.path.join(ann_dir, file)
                with open(ann_file_path, "r", encoding="utf8") as ann_file:
                    annotation = json.load(ann_file)

                for item in annotation["form"]:
                    words = [w for w in item["words"] if w["text"].strip() != ""]
                    if len(words) == 0:
                        continue
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append(ner_tags_to_idx[w["label"]])
                        bboxes.append([x / 1000.0 for x in w["box"]])

                image_path = img_dir / file
                image_path = str(image_path).replace("json", "jpg")
                data.append([image_path, tokens, bboxes, ner_tags])

        # store the data into pd dataframe
        data_columns = [DataKeysEnum.IMAGE_FILE_PATH, "tokens", "bboxes", "ner_tags"]
        return pd.DataFrame(data, columns=data_columns)

    def _tokenize(self, data):
        tokenized_data = DataTokenizer.tokenize_textual_data(
            data["tokens"].to_list(), self.data_args
        )

        new_data = pd.DataFrame()
        new_data["input_ids"] = tokenized_data["input_ids"]
        new_data["attention_mask"] = tokenized_data["attention_mask"]

        image_file_path = []
        bboxes = []
        ner_tags = []
        words_to_token_maps = []
        words = []
        for batch_index in range(len(tokenized_data["input_ids"])):
            word_ids = tokenized_data.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_data["overflow_to_sample_mapping"][batch_index]

            previous_word_idx = None
            words_to_token_map = []
            seq_length = len(word_ids)
            for (idx, word_idx) in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the labels only
                # for our known words since our classifier is word basd
                # We set the label and bounding box for the first token of each word.
                if word_idx is not None:
                    if word_idx != previous_word_idx:
                        words_to_token_map.append([0] * seq_length)
                    words_to_token_map[-1][idx] = 1

                previous_word_idx = word_idx

            image_file_path.append(data[DataKeysEnum.IMAGE_FILE_PATH][org_batch_index])
            words_to_token_maps.append(words_to_token_map)
            valid_word_ids = [word_id for word_id in word_ids if word_id is not None]
            bboxes.append(
                [
                    value
                    for index, value in enumerate(data["bboxes"][org_batch_index])
                    if index in valid_word_ids
                ]
            )
            ner_tags.append(
                [
                    value
                    for index, value in enumerate(data["ner_tags"][org_batch_index])
                    if index in valid_word_ids
                ]
            )
            words.append(
                [
                    value
                    for index, value in enumerate(data["tokens"][org_batch_index])
                    if index in valid_word_ids
                ]
            )

        new_data[DataKeysEnum.IMAGE_FILE_PATH] = image_file_path
        new_data["bboxes"] = bboxes
        new_data["ner_tags"] = ner_tags
        new_data["words_to_token_maps"] = words_to_token_maps
        new_data["words"] = words
        return data

    def get_data_file_name(self):
        file_name = super().get_data_file_name()
        return f"{file_name}.{SroieDataset.scheme}"

    def get_height_and_width(self, idx):
        image_file_path = self.data.iloc[idx][DataKeysEnum.IMAGE_FILE_PATH]
        width, height = Image.open(image_file_path).size
        return height, width
