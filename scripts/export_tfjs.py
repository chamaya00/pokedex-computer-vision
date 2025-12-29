"""
Export trained model to TensorFlow.js format for browser deployment.

This script:
1. Loads the trained Keras model
2. Converts to TensorFlow.js format
3. Applies quantization to reduce model size
4. Copies labels.json for browser use

Usage:
    python scripts/export_tfjs.py
"""

import json
import shutil
import sys
from pathlib import Path


def export_to_tfjs(model_path: Path, output_dir: Path, quantize: bool = True):
    """Export Keras model to TensorFlow.js format."""
    import tensorflow as tf
    import tensorflowjs as tfjs

    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export with quantization
    print(f"Converting to TensorFlow.js format...")
    if quantize:
        tfjs.converters.save_keras_model(
            model,
            str(output_dir),
            quantization_dtype_map={"uint8": "*"},  # Quantize all layers to uint8
        )
    else:
        tfjs.converters.save_keras_model(model, str(output_dir))

    print(f"Model exported to: {output_dir}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("**/*") if f.is_file())
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

    # List exported files
    print("\nExported files:")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")

    return total_size


def copy_labels(models_dir: Path, tfjs_dir: Path):
    """Copy labels.json to TensorFlow.js directory."""
    labels_src = models_dir / "labels.json"
    labels_dst = tfjs_dir / "labels.json"

    if labels_src.exists():
        shutil.copy(labels_src, labels_dst)
        print(f"\nCopied labels.json to {labels_dst}")
    else:
        print(f"\nWarning: labels.json not found at {labels_src}")


def create_zip(tfjs_dir: Path, output_path: Path):
    """Create zip file of TensorFlow.js model."""
    import zipfile

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in tfjs_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(tfjs_dir.parent)
                zipf.write(file, arcname)

    print(f"\nCreated zip: {output_path}")
    print(f"Zip size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.config import MODELS_DIR

    # Check if model exists
    model_path = MODELS_DIR / "best_model.keras"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using the Colab notebook.")
        sys.exit(1)

    # Export to TensorFlow.js
    tfjs_dir = MODELS_DIR / "tfjs_model"
    export_to_tfjs(model_path, tfjs_dir, quantize=True)

    # Copy labels
    copy_labels(MODELS_DIR, tfjs_dir)

    # Create zip for easy download
    zip_path = MODELS_DIR / "tfjs_model.zip"
    create_zip(tfjs_dir, zip_path)

    print("\nExport complete!")
    print(f"Model files: {tfjs_dir}")
    print(f"Zip file: {zip_path}")
