from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import torch


def compute_pos_weights(csv_file: str | Path) -> torch.Tensor:
    """
    Compute stable positive class weights for BCEWithLogitsLoss.

    Original:
        weight = negative / positive

    Improved:
        weight = sqrt(negative / positive)

    Then clip to avoid extremely large gradients.
    """

    csv_file = Path(csv_file)

    df = pd.read_csv(csv_file)

    answer_columns = [
        c
        for c in df.columns
        if c.startswith("answer::")
    ]

    weights = []

    ignored = []
    rare = []
    very_rare = []

    print()
    print("=" * 90)
    print("COMPUTING POSITIVE CLASS WEIGHTS")
    print("=" * 90)

    for column in answer_columns:

        positive = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        negative = len(df) - positive

        # -------------------------------
        # Stable weighting
        # -------------------------------

        if positive == 0:

            weight = 0.0

        else:

            weight = math.sqrt(
                negative / positive
            )

            weight = min(weight, 5.0)

        weights.append(weight)

        print(
            f"{column[:45]:45}"
            f" Pos={positive:4d}"
            f" Neg={negative:4d}"
            f" Weight={weight:.2f}"
        )

        if weight == 0:
            ignored.append(column)

        elif weight >= 5:
            very_rare.append(column)

        elif weight >= 3:
            rare.append(column)

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(f"Labels           : {len(answer_columns)}")
    print(f"Min Weight       : {weights.min().item():.3f}")
    print(f"Max Weight       : {weights.max().item():.3f}")
    print(f"Mean Weight      : {weights.mean().item():.3f}")
    print(f"Median Weight    : {weights.median().item():.3f}")
    print()

    print(f"Ignored Labels   : {len(ignored)}")
    print(f"Rare Labels      : {len(rare)}")
    print(f"Very Rare Labels : {len(very_rare)}")

    if ignored:

        print("\nIgnored:")

        for c in ignored:
            print("  -", c)

    if rare:

        print("\nRare:")

        for c in rare:
            print("  -", c)

    if very_rare:

        print("\nVery Rare:")

        for c in very_rare:
            print("  -", c)

    print("=" * 90)
    print()

    return weights