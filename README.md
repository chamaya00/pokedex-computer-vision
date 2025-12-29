# Pokedex Classifier

A Pokemon image classifier using transfer learning with MobileNetV3. Identifies Gen 1 Pokemon (151 classes) from images.

## Project Structure

```
pokedex-classifier/
├── notebooks/
│   └── pokemon_classifier_training.ipynb  # Main training notebook (Colab)
├── scripts/
│   ├── download_dataset.py                # Download Kaggle dataset
│   ├── prepare_data.py                    # Split into train/val/test
│   ├── evaluate_model.py                  # Benchmark evaluation
│   └── export_tfjs.py                     # Export to TensorFlow.js
├── src/
│   └── config.py                          # Central configuration
├── data/                                  # Dataset (not in git)
├── models/                                # Trained models (not in git)
└── benchmark/                             # Golden test set
```

## Quick Start

### 1. Training in Google Colab

1. Open `notebooks/pokemon_classifier_training.ipynb` in Google Colab
2. Upload your `kaggle.json` API credentials
3. Run all cells
4. Download the exported `tfjs_model.zip`

### 2. Local Setup (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API setup)
python scripts/download_dataset.py

# Prepare data splits
python scripts/prepare_data.py
```

## Model Details

- **Base Model**: MobileNetV3Small (ImageNet pretrained)
- **Input Size**: 224x224 RGB
- **Output**: 151 classes (Gen 1 Pokemon)
- **Export Format**: TensorFlow.js (quantized)
- **Model Size**: ~3-5 MB (quantized)

## Training Approach

1. **Stage 1**: Train classification head with frozen base (10 epochs)
2. **Stage 2**: Fine-tune top 20 layers with lower LR (15 epochs)

## Target Metrics

- Top-1 Accuracy: >80%
- Top-5 Accuracy: >95%
- Inference Time: <100ms (browser)

## Kaggle API Setup

1. Go to kaggle.com -> Account -> Create New API Token
2. Download `kaggle.json`
3. Place in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows)
4. Run `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

## Next Steps

**Day 2: Browser Integration**
- Download tfjs_model.zip
- Create React/Next.js app
- Load model with TensorFlow.js
- Add camera capture
- Build Pokedex UI

**Day 3+: Improvements**
- Create benchmark dataset
- Add Pokemon card images to training data
- Test on real-world photos
- Iterate based on failures
