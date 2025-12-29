"""
Download Pokemon dataset from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to kaggle.com -> Account -> Create New API Token
   - Download kaggle.json
   - Place in ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)
   - chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_dataset.py

For Colab:
    Upload kaggle.json and run the notebook cells
"""

import os
import sys
import zipfile
from pathlib import Path


def download_kaggle_dataset(dataset_id: str, output_dir: Path) -> None:
    """Download and extract dataset from Kaggle."""

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Please install kaggle: pip install kaggle")
        sys.exit(1)

    # Initialize API
    api = KaggleApi()
    api.authenticate()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_id}...")
    api.dataset_download_files(dataset_id, path=output_dir, unzip=True)
    print(f"Dataset downloaded to {output_dir}")


def verify_dataset(data_dir: Path) -> dict:
    """Verify dataset structure and count images per class."""

    # Find the Pokemon images directory
    pokemon_dir = None
    for item in data_dir.rglob("*"):
        if item.is_dir() and any(item.iterdir()):
            # Check if this looks like the Pokemon directory
            subdirs = [d for d in item.iterdir() if d.is_dir()]
            if len(subdirs) > 100:  # Should have ~151 Pokemon folders
                pokemon_dir = item
                break

    if pokemon_dir is None:
        print("Could not find Pokemon image directory!")
        return {}

    print(f"Found Pokemon data at: {pokemon_dir}")

    # Count images per class
    class_counts = {}
    for class_dir in sorted(pokemon_dir.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            class_counts[class_dir.name] = len(images)

    # Print summary
    print(f"\nFound {len(class_counts)} Pokemon classes")
    print(f"Total images: {sum(class_counts.values())}")
    print(f"Min images per class: {min(class_counts.values())}")
    print(f"Max images per class: {max(class_counts.values())}")
    print(f"Avg images per class: {sum(class_counts.values()) / len(class_counts):.1f}")

    # Show classes with fewest images
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    print(f"\nClasses with fewest images:")
    for name, count in sorted_counts[:5]:
        print(f"  {name}: {count}")

    return class_counts


if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.config import DATA_DIR, KAGGLE_DATASET

    download_kaggle_dataset(KAGGLE_DATASET, DATA_DIR)
    verify_dataset(DATA_DIR)
