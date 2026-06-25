import torch
import torch.nn as nn

from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    DeBERTa-v3-Large text encoder.

    Returns the CLS embedding for multimodal fusion.
    """

    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        dropout=0.1,
    ):

        super().__init__()

        print("=" * 60)
        print("Building Text Encoder")
        print("=" * 60)
        print(f"Backbone : {model_name}")

        self.encoder = AutoModel.from_pretrained(
            model_name,
        )

        self.hidden_size = (
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(
            dropout,
        )

        print(
            f"Hidden Size : {self.hidden_size}"
        )

    @property
    def output_dim(self):

        return self.hidden_size

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        outputs = self.encoder(

            input_ids=input_ids,

            attention_mask=attention_mask,

        )

        cls = outputs.last_hidden_state[:, 0]

        cls = self.dropout(
            cls,
        )

        return cls