from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import train_test_split

from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

TARGET_CLASSES = {

    "invoice",

    "form",

    "letter",

    "memo",

    "specification",

    "budget",

}

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

DATASET_ROOT = Path(
    "data/raw/test"
)

OUTPUT_DIR = Path(
    "data/processed/vision"
)

TRAIN_RATIO = 0.80

VAL_RATIO = 0.10

TEST_RATIO = 0.10

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6

random.seed(RANDOM_SEED)

# ============================================================
# Utility Functions
# ============================================================


def print_header(title: str):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)

    print()


def verify_image(path: Path):

    """
    Verify image integrity.

    Returns
    -------
    valid : bool

    width : int

    height : int
    """

    try:

        with Image.open(path) as img:

            img.verify()

        with Image.open(path) as img:

            width, height = img.size

        return True, width, height

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):

        return False, 0, 0


def get_image_files(folder: Path):

    files = []

    for ext in SUPPORTED_EXTENSIONS:

        files.extend(folder.glob(f"*{ext}"))

        files.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(files)


def save_json(obj, path):

    with open(path, "w") as f:

        json.dump(
            obj,
            f,
            indent=4,
        )


def save_csv(df, path):

    df.to_csv(
        path,
        index=False,
    )

# ============================================================
# Scan Dataset
# ============================================================

def scan_dataset():

    print_header("Scanning Dataset")

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_ROOT.resolve()}"
        )

    classes = sorted(

    [

        d.name

        for d in DATASET_ROOT.iterdir()

        if d.is_dir()

        and d.name in TARGET_CLASSES

    ]

)
    print(f"Found {len(classes)} classes\n")

    class_mapping = {

        cls: idx

        for idx, cls in enumerate(classes)

    }

    records = []

    duplicate_paths = set()

    seen_paths = set()

    corrupted_images = []

    class_counter = Counter()

    for class_name in classes:

        class_dir = DATASET_ROOT / class_name

        image_files = get_image_files(class_dir)

        print(

            f"{class_name:<30}"

            f"{len(image_files):>6} images"

        )

        for image_path in tqdm(

            image_files,

            desc=class_name,

            leave=False,

        ):

            path_str = str(image_path.resolve())

            ####################################################
            # Duplicate Detection
            ####################################################

            if path_str in seen_paths:

                duplicate_paths.add(path_str)

                continue

            seen_paths.add(path_str)

            ####################################################
            # Image Validation
            ####################################################

            valid, width, height = verify_image(image_path)

            if not valid:

                corrupted_images.append(path_str)

                continue

            ####################################################
            # Record
            ####################################################

            records.append(

                {

                    "image_path": path_str,

                    "label": class_mapping[class_name],

                    "class_name": class_name,

                    "filename": image_path.name,

                    "extension": image_path.suffix.lower(),

                    "width": width,

                    "height": height,

                }

            )

            class_counter[class_name] += 1

    if len(records) == 0:

        raise RuntimeError(
            "No valid images were found."
        )

    df = pd.DataFrame(records)

    print()

    print("=" * 70)

    print("Scan Summary")

    print("=" * 70)

    print(f"Valid Images      : {len(df)}")

    print(f"Corrupted Images  : {len(corrupted_images)}")

    print(f"Duplicate Images  : {len(duplicate_paths)}")

    print()

    return (

        df,

        class_mapping,

        class_counter,

        corrupted_images,

        duplicate_paths,

    )

# ============================================================
# Stratified Dataset Split
# ============================================================

def split_dataset(df: pd.DataFrame):

    print_header("Creating Stratified Train / Validation / Test Split")

    ####################################################
    # Train Split
    ####################################################

    train_df, temp_df = train_test_split(

        df,

        test_size=(1.0 - TRAIN_RATIO),

        random_state=RANDOM_SEED,

        stratify=df["label"],

    )

    ####################################################
    # Validation/Test Split
    ####################################################

    relative_val_ratio = VAL_RATIO / (
        VAL_RATIO + TEST_RATIO
    )

    val_df, test_df = train_test_split(

        temp_df,

        test_size=(1.0 - relative_val_ratio),

        random_state=RANDOM_SEED,

        stratify=temp_df["label"],

    )

    ####################################################
    # Reset Index
    ####################################################

    train_df = train_df.reset_index(drop=True)

    val_df = val_df.reset_index(drop=True)

    test_df = test_df.reset_index(drop=True)

    ####################################################
    # Statistics
    ####################################################

    print(f"Training Images   : {len(train_df)}")

    print(f"Validation Images : {len(val_df)}")

    print(f"Test Images       : {len(test_df)}")

    print()

    return train_df, val_df, test_df

# ============================================================
# Save Outputs
# ============================================================

def save_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    class_mapping: dict,
    class_counter,
    corrupted_images,
    duplicate_paths,
):

    print_header("Saving Dataset")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ####################################################
    # CSV Files
    ####################################################

    save_csv(
        train_df,
        OUTPUT_DIR / "train.csv",
    )

    save_csv(
        val_df,
        OUTPUT_DIR / "val.csv",
    )

    save_csv(
        test_df,
        OUTPUT_DIR / "test.csv",
    )

    ####################################################
    # Class Mapping
    ####################################################

    save_json(
        class_mapping,
        OUTPUT_DIR / "class_mapping.json",
    )

    ####################################################
    # Dataset Report
    ####################################################

    report = {

        "random_seed": RANDOM_SEED,

        "num_classes": len(class_mapping),

        "total_images": len(train_df)
        + len(val_df)
        + len(test_df),

        "train_images": len(train_df),

        "validation_images": len(val_df),

        "test_images": len(test_df),

        "corrupted_images": len(
            corrupted_images
        ),

        "duplicate_images": len(
            duplicate_paths
        ),

        "class_distribution": dict(
            class_counter
        ),

    }

    save_json(
        report,
        OUTPUT_DIR / "vision_dataset_report.json",
    )

    ####################################################
    # Console Summary
    ####################################################

    print("=" * 70)

    print("Dataset Summary")

    print("=" * 70)

    print(f"Classes             : {len(class_mapping)}")

    print(f"Total Images        : {report['total_images']}")

    print(f"Train Images        : {report['train_images']}")

    print(f"Validation Images   : {report['validation_images']}")

    print(f"Test Images         : {report['test_images']}")

    print(f"Corrupted Images    : {report['corrupted_images']}")

    print(f"Duplicate Images    : {report['duplicate_images']}")

    print()

    print("Class Distribution")

    print("-" * 70)

    for cls, count in sorted(
        class_counter.items()
    ):

        print(
            f"{cls:<35} {count:>6}"
        )

    print()

    print("Saved to")

    print(OUTPUT_DIR.resolve())

# ============================================================
# Main
# ============================================================

def main():

    print_header(
        "Vision Dataset Preparation"
    )

    ####################################################
    # Scan
    ####################################################

    (
        df,
        class_mapping,
        class_counter,
        corrupted_images,
        duplicate_paths,
    ) = scan_dataset()

    ####################################################
    # Integrity Checks
    ####################################################

    print_header(
        "Running Integrity Checks"
    )

    if len(df) == 0:

        raise RuntimeError(
            "Dataset is empty."
        )

    if df["label"].nunique() < 2:

        raise RuntimeError(
            "At least two classes are required."
        )

    if df["image_path"].duplicated().any():

        raise RuntimeError(
            "Duplicate image paths detected."
        )

    missing = [

        p

        for p in df["image_path"]

        if not Path(p).exists()

    ]

    if len(missing):

        raise RuntimeError(

            f"{len(missing)} missing images detected."

        )

    print("✓ Integrity check passed")

    ####################################################
    # Split
    ####################################################

    train_df, val_df, test_df = split_dataset(
        df
    )

    ####################################################
    # Save
    ####################################################

    save_outputs(

        train_df,

        val_df,

        test_df,

        class_mapping,

        class_counter,

        corrupted_images,

        duplicate_paths,

    )

    print()

    print("=" * 70)

    print("Vision Dataset Ready")

    print("=" * 70)

    print()

    print(f"Output directory:")

    print(OUTPUT_DIR.resolve())

    print()

    print("Generated")

    print("  ✓ train.csv")

    print("  ✓ val.csv")

    print("  ✓ test.csv")

    print("  ✓ class_mapping.json")

    print("  ✓ vision_dataset_report.json")


if __name__ == "__main__":

    main()