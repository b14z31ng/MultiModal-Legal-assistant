import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(
    labels,
    logits,
    threshold=0.5,
):

    probabilities = 1.0 / (
        1.0 + np.exp(-logits)
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        labels,
        predictions,
    )

    metrics["precision"] = precision_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    metrics["f1"] = f1_score(
        labels,
        predictions,
        average="micro",
        zero_division=0,
    )

    return metrics


def search_best_threshold(
    labels,
    logits,
):

    probabilities = 1.0 / (
        1.0 + np.exp(-logits)
    )

    best_threshold = 0.50
    best_f1 = -1.0

    history = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.05,
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            labels,
            predictions,
            average="micro",
            zero_division=0,
        )

        history.append(
            (
                threshold,
                f1,
            )
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    print()
    print("=" * 70)
    print("THRESHOLD SEARCH")
    print("=" * 70)

    for threshold, f1 in history:

        print(
            f"{threshold:.2f}"
            f" -> "
            f"{f1:.4f}"
        )

    print("=" * 70)
    print(
        f"BEST THRESHOLD : {best_threshold:.2f}"
    )
    print(
        f"BEST F1        : {best_f1:.4f}"
    )
    print("=" * 70)
    print()

    return {

    "threshold":

        best_threshold,

    "f1":

        best_f1,

}