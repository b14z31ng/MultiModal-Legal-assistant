import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report,

    top_k_accuracy_score,

)


def compute_metrics(

    labels,

    logits,

):

    predictions = np.argmax(

        logits,

        axis=1,

    )

    metrics = {}

    ####################################################
    # Accuracy
    ####################################################

    metrics["accuracy"] = accuracy_score(

        labels,

        predictions,

    )

    ####################################################
    # Top-3
    ####################################################

    metrics["top3_accuracy"] = top_k_accuracy_score(

        labels,

        logits,

        k=3,

        labels=list(range(logits.shape[1])),

    )

    ####################################################
    # Top-5
    ####################################################

    metrics["top5_accuracy"] = top_k_accuracy_score(

        labels,

        logits,

        k=5,

        labels=list(range(logits.shape[1])),

    )

    ####################################################
    # Precision
    ####################################################

    metrics["precision"] = precision_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ####################################################
    # Recall
    ####################################################

    metrics["recall"] = recall_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ####################################################
    # F1
    ####################################################

    metrics["f1"] = f1_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ####################################################
    # Confusion Matrix
    ####################################################

    metrics["confusion_matrix"] = confusion_matrix(

        labels,

        predictions,

    )

    ####################################################
    # Report
    ####################################################

    metrics["classification_report"] = classification_report(

        labels,

        predictions,

        output_dict=True,

        zero_division=0,

    )

    return metrics