import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    average_precision_score,

    confusion_matrix,

    classification_report,

    hamming_loss,

    jaccard_score,

    top_k_accuracy_score,

)


############################################################
# Vision Metrics
############################################################

def compute_vision_metrics(

    labels,

    logits,

):

    predictions = np.argmax(

        logits,

        axis=1,

    )

    metrics = {}

    ########################################################

    metrics["accuracy"] = accuracy_score(

        labels,

        predictions,

    )

    ########################################################

    metrics["top3_accuracy"] = top_k_accuracy_score(

        labels,

        logits,

        k=3,

    )

    ########################################################

    metrics["top5_accuracy"] = top_k_accuracy_score(

        labels,

        logits,

        k=5,

    )

    ########################################################

    metrics["precision"] = precision_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ########################################################

    metrics["recall"] = recall_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ########################################################

    metrics["f1"] = f1_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ########################################################

    metrics["confusion_matrix"] = confusion_matrix(

        labels,

        predictions,

    )

    ########################################################

    metrics["classification_report"] = classification_report(

        labels,

        predictions,

        output_dict=True,

        zero_division=0,

    )

    return metrics


############################################################
# Multimodal Metrics
############################################################

def compute_multilabel_metrics(

    labels,

    logits,

    threshold=0.5,

):

    probabilities = 1.0 / (

        1.0 +

        np.exp(

            -logits

        )

    )

    predictions = (

        probabilities >= threshold

    ).astype(np.int32)

    labels = labels.astype(np.int32)

    metrics = {}

    ########################################################
    # Exact Match
    ########################################################

    metrics["subset_accuracy"] = np.mean(

        np.all(

            labels == predictions,

            axis=1,

        )

    )

    ########################################################
    # Label Accuracy
    ########################################################

    metrics["label_accuracy"] = np.mean(

        labels == predictions

    )

    ########################################################

    metrics["precision"] = precision_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ########################################################

    metrics["recall"] = recall_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ########################################################

    metrics["f1"] = f1_score(

        labels,

        predictions,

        average="macro",

        zero_division=0,

    )

    ########################################################

    metrics["hamming_loss"] = hamming_loss(

        labels,

        predictions,

    )

    ########################################################

    metrics["jaccard"] = jaccard_score(

        labels,

        predictions,

        average="samples",

        zero_division=0,

    )

    ########################################################

    try:

        metrics["roc_auc"] = roc_auc_score(

            labels,

            probabilities,

            average="macro",

        )

    except Exception:

        metrics["roc_auc"] = None

    ########################################################

    try:

        metrics["pr_auc"] = average_precision_score(

            labels,

            probabilities,

            average="macro",

        )

    except Exception:

        metrics["pr_auc"] = None

    return metrics