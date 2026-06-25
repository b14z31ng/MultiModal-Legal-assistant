import torch.nn as nn


class RiskClassifier(
    nn.Module
):

    def __init__(

        self,

        fusion_dim,

        hidden_dim,

        num_labels,

        dropout,

    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_dim,
                num_labels,
            ),

        )

    def forward(
        self,
        x,
    ):

        return self.classifier(
            x
        )