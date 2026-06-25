from src.constants import (
    HIGH_RISK_CLAUSES,
)


def build_binary_label(
    sample,
):

    category = sample.get(
        "category",
        "",
    )

    return int(
        category in HIGH_RISK_CLAUSES
    )