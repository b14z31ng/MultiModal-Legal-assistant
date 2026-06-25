from pathlib import Path

import mlflow


def initialize_mlflow(
    experiment_name,
):
    """
    Initialize MLflow using SQLite backend.
    """

    database = Path(
        "mlflow.db"
    ).resolve()

    tracking_uri = (
        f"sqlite:///{database}"
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        experiment_name
    )


def log_parameters(
    params,
):

    if params is None:
        return

    if hasattr(
        params,
        "to_dict",
    ):

        params = params.to_dict()

    for key, value in params.items():

        try:

            mlflow.log_param(
                key,
                value,
            )

        except Exception:

            pass


def log_metrics(
    metrics,
    step,
):

    for key, value in metrics.items():

        if key == "confusion_matrix":

            continue

        if value is None:

            continue

        try:

            mlflow.log_metric(
                key,
                float(value),
                step=step,
            )

        except Exception:

            pass


def log_artifact(
    artifact_path,
):

    artifact_path = Path(
        artifact_path
    )

    if artifact_path.exists():

        mlflow.log_artifact(
            str(artifact_path)
        )


def log_directory(
    directory,
):

    directory = Path(
        directory
    )

    if directory.exists():

        mlflow.log_artifacts(
            str(directory)
        )