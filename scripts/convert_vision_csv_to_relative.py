from pathlib import Path

import pandas as pd

# ============================================================
# Configuration
# ============================================================

VISION_DIR = Path("data/processed/vision")

CSV_FILES = [
    "train.csv",
    "val.csv",
    "test.csv",
]

# Current absolute root on your laptop
LOCAL_ROOT = Path("data/raw").resolve()

print("=" * 70)
print("Converting Vision CSV Paths")
print("=" * 70)

for csv_name in CSV_FILES:

    csv_path = VISION_DIR / csv_name

    df = pd.read_csv(csv_path)

    new_paths = []

    for path in df["image_path"]:

        p = Path(path).resolve()

        try:
            # Example:
            # /home/.../data/raw/test/form/a.tif
            # ->
            # test/form/a.tif
            rel = p.relative_to(LOCAL_ROOT)

        except ValueError:
            raise RuntimeError(
                f"\n{p}\n"
                f"is not inside\n"
                f"{LOCAL_ROOT}"
            )

        new_paths.append(
            rel.as_posix()
        )

    df["image_path"] = new_paths

    df.to_csv(
        csv_path,
        index=False,
    )

    print(f"✓ {csv_name}")

print()
print("=" * 70)
print("Finished Successfully")
print("=" * 70)