from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)


def save_confusion_matrix(
    confusion_matrix,
    class_names,
    save_path,
):
    """
    Save confusion matrix figure.
    """

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 12)
    )

    display = ConfusionMatrixDisplay(

        confusion_matrix=confusion_matrix,

        display_labels=class_names,

    )

    display.plot(

        cmap="Blues",

        xticks_rotation=90,

        ax=ax,

        colorbar=False,

    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close(fig)


def save_classification_report(
    labels,
    predictions,
    class_names,
    save_path,
):
    """
    Save sklearn classification report.
    """

    report = classification_report(

        labels,

        predictions,

        target_names=class_names,

        output_dict=True,

        zero_division=0,

    )

    save_path = Path(save_path)

    save_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(

        save_path,

        "w",

    ) as f:

        json.dump(

            report,

            f,

            indent=4,

        )


def save_metrics(
    metrics,
    save_path,
):
    """
    Save metrics dictionary.
    """

    metrics = {

        k: (

            float(v)

            if isinstance(
                v,
                np.generic,
            )

            else v

        )

        for k, v in metrics.items()

        if k != "confusion_matrix"

    }

    save_path = Path(save_path)

    save_path.parent.mkdir(

        parents=True,

        exist_ok=True,

    )

    with open(

        save_path,

        "w",

    ) as f:

        json.dump(

            metrics,

            f,

            indent=4,

        )