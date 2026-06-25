from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

CUAD_JSON = ROOT / "data" / "raw" / "cuad-main" / "data" / "CUADv1.json"

OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json():

    with open(CUAD_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(data):

    clause_names = []

    rows = []

    print("\nProcessing contracts...\n")

    for contract in tqdm(data["data"]):

        title = contract["title"]

        paragraph = contract["paragraphs"][0]

        context = paragraph["context"]

        label_vector = {}

        answer_texts = {}

        for qa in paragraph["qas"]:

            clause = qa["question"]

            clause_names.append(clause)

            present = 0 if qa["is_impossible"] else 1

            label_vector[clause] = present

            if present:

                answer_texts[clause] = " ".join(
                    ans["text"] for ans in qa["answers"]
                )

            else:

                answer_texts[clause] = ""

        row = {
            "title": title,
            "contract_text": context,
        }

        for clause in sorted(set(clause_names)):

            row[f"label::{clause}"] = label_vector.get(
                clause,
                0,
            )

            row[f"answer::{clause}"] = answer_texts.get(
                clause,
                "",
            )

        rows.append(row)

    clause_names = sorted(set(clause_names))

    return pd.DataFrame(rows), clause_names


def save_mapping(clause_names):

    mapping = {
        idx: name
        for idx, name in enumerate(clause_names)
    }

    with open(
        OUTPUT_DIR / "clause_mapping.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            mapping,
            f,
            indent=4,
            ensure_ascii=False,
        )


def split(df):

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    train, val = train_test_split(
        train,
        test_size=0.1,
        random_state=42,
        shuffle=True,
    )

    return train, val, test


def main():

    print("=" * 60)
    print("Preparing CUAD Dataset")
    print("=" * 60)

    data = load_json()

    print(f"\nContracts : {len(data['data'])}")

    df, clause_names = build_dataset(data)

    save_mapping(clause_names)

    train, val, test = split(df)

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    val.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    df.to_csv(
        OUTPUT_DIR / "cuad.csv",
        index=False,
    )

    print("\nFinished Successfully")

    print(f"\nTotal Contracts : {len(df)}")

    print(f"Train : {len(train)}")

    print(f"Validation : {len(val)}")

    print(f"Test : {len(test)}")

    print(f"\nClause Types : {len(clause_names)}")

    print("\nSaved to")

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()