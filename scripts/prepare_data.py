"""
Prepare dataset for training by splitting into train/val/test sets.

This script:
1. Finds all Pokemon images
2. Splits into train/val/test with stratification
3. Creates the required directory structure
4. Generates class labels JSON

Usage:
    python scripts/prepare_data.py [--source official|kaggle|augmented]

Options:
    --source: Which dataset to use
        - kaggle: Original Kaggle dataset (default)
        - official: Official sprites from PokeAPI
        - augmented: Augmented official sprites (recommended for training)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from collections import defaultdict
import random


def find_pokemon_directory(data_dir: Path, source: str = "kaggle") -> Path:
    """
    Find the directory containing Pokemon class folders.

    Args:
        data_dir: Base data directory
        source: Dataset source ('kaggle', 'official', or 'augmented')

    Returns:
        Path to the Pokemon images directory
    """
    # Check for specific source directories first
    if source == "official":
        official_dir = data_dir / "pokemon_official"
        if official_dir.exists():
            return official_dir
    elif source == "augmented":
        augmented_dir = data_dir / "pokemon_augmented"
        if augmented_dir.exists():
            return augmented_dir

    # Fall back to auto-detection for kaggle or if specific dir not found
    for item in data_dir.rglob("*"):
        if item.is_dir():
            subdirs = [d for d in item.iterdir() if d.is_dir()]
            if len(subdirs) > 100:
                return item

    raise FileNotFoundError(f"Could not find Pokemon image directory for source '{source}'")


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> dict:
    """Split dataset into train/val/test with stratification."""

    random.seed(seed)

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all images by class
    class_images = defaultdict(list)
    for class_dir in sorted(source_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name.lower().replace(" ", "_")
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            class_images[class_name] = images

    # Split each class
    split_counts = {"train": 0, "val": 0, "test": 0}
    class_names = []

    for class_name, images in sorted(class_images.items()):
        class_names.append(class_name)

        # Shuffle images
        random.shuffle(images)

        # Calculate split indices
        n = len(images)
        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        n_train = n - n_test - n_val

        # Split
        train_images = images[:n_train]
        val_images = images[n_train : n_train + n_val]
        test_images = images[n_train + n_val :]

        # Create class directories and copy images
        for split_name, split_images in [
            ("train", train_images),
            ("val", val_images),
            ("test", test_images),
        ]:
            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for i, img_path in enumerate(split_images):
                ext = img_path.suffix
                new_name = f"{class_name}_{i:04d}{ext}"
                shutil.copy(img_path, split_class_dir / new_name)
                split_counts[split_name] += 1

    # Save class names mapping
    labels_path = output_dir / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(
            {
                "class_names": class_names,
                "num_classes": len(class_names),
                "index_to_name": {i: name for i, name in enumerate(class_names)},
                "name_to_index": {name: i for i, name in enumerate(class_names)},
            },
            f,
            indent=2,
        )

    print(f"Dataset split complete:")
    print(f"  Train: {split_counts['train']} images")
    print(f"  Val: {split_counts['val']} images")
    print(f"  Test: {split_counts['test']} images")
    print(f"  Classes: {len(class_names)}")
    print(f"  Labels saved to: {labels_path}")

    return {
        "counts": split_counts,
        "class_names": class_names,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.config import DATA_DIR, VALIDATION_SPLIT, TEST_SPLIT

    # Parse arguments
    parser = argparse.ArgumentParser(description="Prepare Pokemon dataset for training")
    parser.add_argument(
        "--source",
        choices=["kaggle", "official", "augmented"],
        default="kaggle",
        help="Dataset source: kaggle (default), official (PokeAPI), or augmented"
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for output directory (e.g., 'split_official')"
    )
    args = parser.parse_args()

    # Find source directory
    source_dir = find_pokemon_directory(DATA_DIR, source=args.source)
    print(f"Using {args.source} dataset")
    print(f"Found Pokemon images at: {source_dir}")

    # Determine output directory
    if args.output_suffix:
        output_dir = DATA_DIR / f"split_{args.output_suffix}"
    elif args.source != "kaggle":
        output_dir = DATA_DIR / f"split_{args.source}"
    else:
        output_dir = DATA_DIR / "split"

    print(f"Output directory: {output_dir}")

    # Split dataset
    split_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        val_split=VALIDATION_SPLIT,
        test_split=TEST_SPLIT,
    )
