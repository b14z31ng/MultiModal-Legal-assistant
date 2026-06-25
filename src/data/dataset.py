from pathlib import Path

import pandas as pd

import torch

from PIL import Image

from torch.utils.data import Dataset

from transformers import AutoTokenizer

from src.data.transforms import build_image_transform


class MultimodalRiskDataset(Dataset):
    """
    Multimodal CUAD Dataset

    Returns

        image

        text

        multilabel target
    """

    def __init__(

        self,

        csv_path,

        image_root,

        tokenizer_name,

        image_size,

        max_length,

        vision_enabled=True,

    ):

        self.df = pd.read_csv(
            csv_path
        )

        self.image_root = Path(
            image_root
        )

        self.max_length = max_length

        self.vision_enabled = vision_enabled

        ####################################################
        # Image
        ####################################################

        self.transform = build_image_transform(
            image_size=image_size,
            train=True,
        )

        ####################################################
        # Tokenizer
        ####################################################

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
        )

        ####################################################
        # Labels
        ####################################################

        self.label_columns = [

            c

            for c in self.df.columns

            if c.startswith(
                "label::"
            )

        ]

    def __len__(self):

        return len(
            self.df
        )

    def _load_image(
        self,
        title,
    ):

        if not self.vision_enabled:

            return None

        image_path = (

            self.image_root

            /

            f"{title}.png"

        )

        if image_path.exists():

            image = (

                Image.open(
                    image_path
                )

                .convert("RGB")

            )

        else:

            image = Image.new(

                "RGB",

                (

                    384,

                    384,

                ),

                "white",

            )

        return self.transform(
            image
        )

    def __getitem__(
        self,
        idx,
    ):

        row = self.df.iloc[
            idx
        ]

        ####################################################
        # Image
        ####################################################

        image = self._load_image(
            row["title"]
        )

        ####################################################
        # Text
        ####################################################

        encoding = self.tokenizer(

            row["contract_text"],

            max_length=self.max_length,

            truncation=True,

            padding="max_length",

            return_tensors="pt",

        )

        ####################################################
        # Labels
        ####################################################

        labels = torch.tensor(

            row[
                self.label_columns
            ].values,

            dtype=torch.float32,

        )

        return {

            "pixel_values":

                image,

            "input_ids":

                encoding[
                    "input_ids"
                ].squeeze(0),

            "attention_mask":

                encoding[
                    "attention_mask"
                ].squeeze(0),

            "labels":

                labels,

        }