from pathlib import Path

import pandas as pd

import torch

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms

from src.data.text_processor import TextProcessor


class MultimodalDataset(Dataset):

    """
    Image + Text dataset.

    Returns

    image

    input_ids

    attention_mask

    label
    """

    def __init__(

        self,

        csv_path,

        image_root,

        tokenizer_name,

        image_size=384,

        max_length=512,

        train=True,

    ):

        self.df = pd.read_csv(csv_path)

        self.image_root = Path(image_root)

        self.processor = TextProcessor(

            tokenizer_name,

            max_length,

        )

        self.train = train

        if train:

            self.transform = transforms.Compose([

                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.RandomRotation(5),

                transforms.RandomAffine(

                    degrees=0,

                    translate=(0.02, 0.02),

                    scale=(0.95, 1.05),

                ),

                transforms.ColorJitter(

                    brightness=0.10,

                    contrast=0.10,

                ),

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=[0.485, 0.456, 0.406],

                    std=[0.229, 0.224, 0.225],

                ),

            ])

        else:

            self.transform = transforms.Compose([

                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=[0.485, 0.456, 0.406],

                    std=[0.229, 0.224, 0.225],

                ),

            ])

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(

            self.image_root / row["image_path"]

        ).convert("RGB")

        image = self.transform(image)

        text = str(row["text"])

        encoded = self.processor(text)

        label = torch.tensor(

            row["label"],

            dtype=torch.long,

        )

        return {

            "pixel_values": image,

            "input_ids":
                encoded["input_ids"],

            "attention_mask":
                encoded["attention_mask"],

            "label": label,

        }