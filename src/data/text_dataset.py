from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CUADTextDataset(Dataset):
    """
    CUAD Multi-label Dataset

    Input:
        Contract text

    Output:
        input_ids
        attention_mask
        labels (41-dimensional multi-label vector)
    """

    def __init__(
        self,
        csv_file,
        tokenizer_name="answerdotai/ModernBERT-base",
        max_length=512,
    ):

        self.csv_file = Path(csv_file)

        self.df = pd.read_csv(self.csv_file)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )

        self.max_length = max_length

        # Automatically detect all answer columns
        self.answer_columns = [

            column

            for column in self.df.columns

            if column.startswith("answer::")

        ]

        self.num_labels = len(self.answer_columns)

        print()
        print("=" * 80)
        print("CUAD TEXT DATASET")
        print("=" * 80)
        print(f"CSV            : {self.csv_file}")
        print(f"Samples        : {len(self.df)}")
        print(f"Answer Columns : {self.num_labels}")
        print("=" * 80)
        print()

    def __len__(self):

        return len(self.df)

    def build_labels(self, row):

        labels = []

        for column in self.answer_columns:

            value = row[column]

            if pd.isna(value):

                labels.append(0.0)

            elif str(value).strip() == "":

                labels.append(0.0)

            else:

                labels.append(1.0)

        return torch.tensor(
            labels,
            dtype=torch.float32,
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        text = str(row["contract_text"])

        encoding = self.tokenizer(

            text,

            truncation=True,

            padding="max_length",

            max_length=self.max_length,

            return_attention_mask=True,

            return_tensors="pt",

        )

        labels = self.build_labels(row)

        return {

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                labels,

        }