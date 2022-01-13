"""
Defines the Funsd dataset.
"""

import json
import os

import pandas as pd
from das.data.datasets.image_dataset_base import ImageDatasetsBase
from das.data.datasets.utils import DataKeysEnum, normalize_bbox
from das.data.tokenizer import DataTokenizer
from das.utils.basic_utils import create_logger
from PIL import Image

logger = create_logger(__name__)


class FunsdDataset(ImageDatasetsBase):
    """FUNSD dataset from https://guillaumejaume.github.io/FUNSD/."""

    is_downloadable = False

    # define dataset labels
    scheme = "IOB"
    NER_TAGS_IOB = [
        "O",
        "B-HEADER",
        "I-HEADER",
        "B-QUESTION",
        "I-QUESTION",
        "B-ANSWER",
        "I-ANSWER",
    ]

    @staticmethod
    def get_ner_tags():
        if FunsdDataset.scheme == "IOB":
            return FunsdDataset.NER_TAGS_IOB
        else:
            raise ValueError("NER scheme not supported!")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_dataset(self):
        # get split dataset dir
        if self.split == "train" or self.split == "val":
            data_dir = self.root_dir / "training_data"
        elif self.split == "test":
            data_dir = self.root_dir / "testing_data"
        else:
            raise ValueError(f"Split argument '{self.split}' not supported.")

        if not data_dir.exists():
            raise ValueError(
                f"Could not find {str(data_dir)} directory in the dataset "
                f"directory: {self.root_dir}"
            )

        # get the length of dataset by getting number of files in images dir
        ann_dir = data_dir / "annotations"
        img_dir = data_dir / "images"

        # load the annotations data from json files for all images
        data = []
        ner_tags = self.get_ner_tags()
        ner_tags_to_idx = dict(zip(ner_tags, range(len(ner_tags))))
        for file in sorted(os.listdir(ann_dir)):
            # load associated annotations
            tokens = []
            bboxes = []
            ner_tags = []

            ann_file_path = os.path.join(ann_dir, file)
            with open(ann_file_path, "r", encoding="utf8") as ann_file:
                annotation = json.load(ann_file)

            image_path = img_dir / file
            image_path = str(image_path).replace("json", "png")
            image_size = Image.open(image_path).size

            # get annotations
            for item in annotation["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue

                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append(ner_tags_to_idx["O"])
                        bboxes.append(normalize_bbox(w["box"], image_size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append(ner_tags_to_idx["B-" + label.upper()])
                    bboxes.append(normalize_bbox(words[0]["box"], image_size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append(ner_tags_to_idx["I-" + label.upper()])
                        bboxes.append(normalize_bbox(w["box"], image_size))

            data.append([image_path, tokens, bboxes, ner_tags])

        # store the data into pd dataframe
        data_columns = [
            DataKeysEnum.IMAGE_FILE_PATH,
            DataKeysEnum.WORDS,
            DataKeysEnum.WORD_BBOXES,
            DataKeysEnum.NER_TAGS,
        ]
        data = pd.DataFrame(data, columns=data_columns)
        return data

    def _tokenize(self, data):
        # get token padding configuration
        padding = "max_length" if self.data_args.pad_to_max_length else False
        tokenizer = DataTokenizer.get_tokenizer(self.data_args)
        tokenized_data = tokenizer(
            text=data["tokens"].to_list(),
            boxes=data["bboxes"].to_list(),
            word_labels=data["ner_tags"].to_list(),
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
        )

        agg_data = pd.DataFrame()
        for (k, v) in tokenized_data.items():
            agg_data[k] = v
        agg_data = agg_data.rename(
            columns={"bbox": "token_bboxes", "labels": "token_ner_tags"}
        )
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
            words_to_token_maps.append(words_to_token_map)

        agg_data[DataKeysEnum.IMAGE_FILE_PATH] = image_file_path
        agg_data["bboxes"] = bboxes
        agg_data["ner_tags"] = ner_tags
        agg_data["words_to_token_maps"] = words_to_token_maps
        agg_data["words"] = words

        return agg_data
