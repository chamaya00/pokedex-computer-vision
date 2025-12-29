"""
Central configuration for the Pokemon classifier project.
All hyperparameters and paths defined here for easy experimentation.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

# For Colab - these will be overridden
COLAB_DATA_DIR = "/content/drive/MyDrive/pokedex/data"
COLAB_MODELS_DIR = "/content/drive/MyDrive/pokedex/models"

# Dataset
KAGGLE_DATASET = "lantian773030/pokemonclassification"  # Gen 1 Pokemon dataset
NUM_CLASSES = 151  # Gen 1 Pokemon

# Image settings
IMAGE_SIZE = 224
CHANNELS = 3

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS_FROZEN = 10  # Train only classification head
EPOCHS_UNFROZEN = 15  # Fine-tune with unfrozen layers
LEARNING_RATE_FROZEN = 1e-3
LEARNING_RATE_UNFROZEN = 1e-5
DROPOUT_RATE = 0.5
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Augmentation settings
ROTATION_RANGE = 20
ZOOM_RANGE = 0.15
BRIGHTNESS_RANGE = (0.8, 1.2)
HORIZONTAL_FLIP = True

# Training settings
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# Model settings
BASE_MODEL = "MobileNetV3Small"  # Options: MobileNetV3Small, MobileNetV3Large, EfficientNetB0
UNFREEZE_LAYERS = 20  # Number of layers to unfreeze during fine-tuning

# Class names will be populated from dataset
CLASS_NAMES = []
