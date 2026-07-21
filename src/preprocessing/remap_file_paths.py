"""
Data path remapping stage.

Purpose:
    Replace Windows absolute paths in docs/file_list.csv
    with valid local Mac paths, while preserving:
    - the exact same rows
    - labels
    - train/val/test assignments
    - row order

This script does not create a new random split.
"""

from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "docs" / "file_list.csv"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "file_list.csv"


def locate_local_file(
    old_path: str,
    dataset_root: Path,
) -> Path:
    """
    Reconstruct the local path from the useful tail
    of the original Windows path.
    """

    windows_path = PureWindowsPath(old_path)
    parts = list(windows_path.parts)

    try:
        dataset_index = next(
            index
            for index, part in enumerate(parts)
            if part.startswith("Fruits_Vegetables_Dataset")
        )
    except StopIteration as error:
        raise ValueError(
            f"Cannot identify dataset root in path: {old_path}"
        ) from error

    relative_parts = parts[dataset_index + 1 :]
    candidate = dataset_root.joinpath(*relative_parts)

    if candidate.is_file():
        return candidate.resolve()

    filename = windows_path.name
    matches = list(dataset_root.rglob(filename))

    if len(matches) == 1:
        return matches[0].resolve()

    if not matches:
        raise FileNotFoundError(
            f"No local match found for: {old_path}"
        )

    raise RuntimeError(
        f"Multiple local matches found for {filename}: "
        f"{[str(path) for path in matches[:5]]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Local root folder of Fruits_Vegetables_Dataset(12000)",
    )
    arguments = parser.parse_args()

    dataset_root = Path(arguments.dataset_root).expanduser().resolve()

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist: {dataset_root}"
        )

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {"filepath", "label", "split"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    original_labels = dataframe["label"].copy()
    original_splits = dataframe["split"].copy()
    original_row_count = len(dataframe)

    new_paths = []

    for row_number, old_path in enumerate(
        dataframe["filepath"],
        start=1,
    ):
        new_path = locate_local_file(
            old_path=str(old_path),
            dataset_root=dataset_root,
        )
        new_paths.append(str(new_path))

        if row_number % 1000 == 0:
            print(
                f"Mapped {row_number}/{original_row_count} paths"
            )

    dataframe["filepath"] = new_paths

    assert len(dataframe) == original_row_count
    assert dataframe["label"].equals(original_labels)
    assert dataframe["split"].equals(original_splits)
    assert dataframe["filepath"].is_unique

    missing_after_mapping = [
        path
        for path in dataframe["filepath"]
        if not Path(path).is_file()
    ]

    if missing_after_mapping:
        raise RuntimeError(
            f"{len(missing_after_mapping)} paths are still missing."
        )

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nUpdated: {OUTPUT_PATH}")
    print(f"Rows preserved: {len(dataframe)}")
    print("Labels preserved: yes")
    print("Splits preserved: yes")
    print("Row order preserved: yes")
    print("A new random split was NOT created.")


if __name__ == "__main__":
    main()