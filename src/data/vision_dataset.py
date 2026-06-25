from pathlib import Path

import pandas as pd

from PIL import Image

import torch

from torch.utils.data import Dataset

from torchvision import transforms


class VisionDataset(Dataset):
    """
    Stage-1 Document Classification Dataset

    Supports:

    • ConvNeXt
    • 384x384
    • Train augmentation
    • Validation transforms
    """

    def __init__(

        self,

        csv_path,

        image_root,

        image_size=384,

        train=True,

    ):

        self.df = pd.read_csv(
            csv_path
        )
        print("=" * 80)
        print("UNIQUE LABELS")
        print(sorted(self.df["label"].unique()))
        print("NUM CLASSES")
        print(len(self.df["label"].unique()))
        print("=" * 80)

        self.image_root = Path(image_root)

        self.train = train
        self.image_size = image_size

        ####################################################
        # Train Transform
        ####################################################

        if train:

            self.transform = transforms.Compose([

                transforms.Resize(

                    (

                        image_size,

                        image_size,

                    )

                ),

                transforms.RandomRotation(

                    degrees=5,

                ),

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

                    mean=[

                        0.485,

                        0.456,

                        0.406,

                    ],

                    std=[

                        0.229,

                        0.224,

                        0.225,

                    ],

                ),

            ])

        ####################################################
        # Validation Transform
        ####################################################

        else:

            self.transform = transforms.Compose([

                transforms.Resize(

                    (

                        image_size,

                        image_size,

                    )

                ),

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=[

                        0.485,

                        0.456,

                        0.406,

                    ],

                    std=[

                        0.229,

                        0.224,

                        0.225,

                    ],

                ),

            ])

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.image_root / row["image_path"]
        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            import os
            import traceback

            print("=" * 80)
            print("PID:", os.getpid())
            print("DATASET OBJECT:", id(self))
            print("IMAGE ROOT:", self.image_root)
            print("IMAGE:", image_path)
            print("EXCEPTION:", repr(e))
            traceback.print_exc()
            print("=" * 80)

            image = Image.new(
                "RGB",
                (self.image_size, self.image_size),
                "white",
            )

        image = self.transform(image)

        label = torch.tensor(
            row["label"],
            dtype=torch.long,
        )
        return {

            "pixel_values": image,

            "label": label,

            "class_name": row["class_name"],

            "image_path": row["image_path"],

        }