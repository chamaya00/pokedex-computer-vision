"""
Evaluate trained model on test set and generate metrics.

This script:
1. Loads the trained model
2. Evaluates on test set
3. Generates confusion matrix
4. Creates classification report
5. Saves misclassified examples

Usage:
    python scripts/evaluate_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_model(model_path: Path):
    """Load trained Keras model."""
    import tensorflow as tf

    return tf.keras.models.load_model(str(model_path))


def create_test_generator(test_dir: Path, image_size: int, batch_size: int):
    """Create test data generator."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return test_generator


def evaluate_model(model, test_generator, output_dir: Path):
    """Evaluate model and generate metrics."""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    # Get predictions
    print("Generating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Get class names
    class_indices = test_generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    class_names = [index_to_class[i] for i in range(len(class_indices))]

    # Calculate metrics
    print("\nEvaluating model...")
    results = model.evaluate(test_generator, verbose=1)
    print(f"\nTest Results:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Top-1 Accuracy: {results[1]:.4f} ({results[1]*100:.1f}%)")
    if len(results) > 2:
        print(f"  Top-5 Accuracy: {results[2]:.4f} ({results[2]*100:.1f}%)")

    # Classification report
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        output_dict=True,
    )

    # Save report
    report_path = output_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nClassification report saved to: {report_path}")

    # Find worst performing classes
    class_f1_scores = {
        k: v["f1-score"]
        for k, v in report.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }
    worst_classes = sorted(class_f1_scores.items(), key=lambda x: x[1])[:10]

    print("\nWorst performing Pokemon:")
    for name, f1 in worst_classes:
        print(f"  {name}: F1={f1:.3f}")

    # Confusion matrix for worst classes
    worst_indices = [class_indices[name] for name, _ in worst_classes]
    mask = np.isin(true_classes, worst_indices)
    cm_subset = confusion_matrix(
        true_classes[mask],
        predicted_classes[mask],
        labels=worst_indices,
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_subset,
        annot=True,
        fmt="d",
        xticklabels=[worst_classes[i][0][:10] for i in range(len(worst_classes))],
        yticklabels=[worst_classes[i][0][:10] for i in range(len(worst_classes))],
        cmap="Blues",
    )
    plt.title("Confusion Matrix (Worst Performing Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(output_dir / "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved to: {output_dir / 'confusion_matrix.png'}")

    # Find misclassified examples
    misclassified_indices = np.where(predicted_classes != true_classes)[0]
    print(f"\nMisclassified examples: {len(misclassified_indices)}/{len(true_classes)}")

    return {
        "accuracy": results[1],
        "loss": results[0],
        "misclassified_count": len(misclassified_indices),
        "total_samples": len(true_classes),
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.config import DATA_DIR, MODELS_DIR, IMAGE_SIZE, BATCH_SIZE

    # Check if model exists
    model_path = MODELS_DIR / "best_model.keras"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using the Colab notebook.")
        sys.exit(1)

    # Check if test data exists
    test_dir = DATA_DIR / "split" / "test"
    if not test_dir.exists():
        print(f"Test data not found at {test_dir}")
        print("Please run prepare_data.py first.")
        sys.exit(1)

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Create test generator
    test_generator = create_test_generator(test_dir, IMAGE_SIZE, BATCH_SIZE)

    # Evaluate
    results = evaluate_model(model, test_generator, MODELS_DIR)

    # Check target
    if results["accuracy"] >= 0.80:
        print(f"\nSUCCESS! Achieved {results['accuracy']*100:.1f}% accuracy (target: 80%)")
    else:
        print(f"\nBelow target. Achieved {results['accuracy']*100:.1f}% (target: 80%)")
