# Diabetic Retinopathy Classification using Ensemble Learning

A deep learning project for automated diabetic retinopathy detection and grading using ensemble methods combining ResNet50, VGG19, and SVM with a meta-learner approach.

## Overview

This project implements a stacked ensemble learning approach to classify diabetic retinopathy severity levels from retinal fundus images. The system achieves **75.8% accuracy** on the test set by combining multiple deep learning models with traditional machine learning classifiers.

## Model Architecture

The ensemble consists of three main components:

1. **ResNet50**: Fine-tuned convolutional neural network with custom classification head
2. **VGG19**: Feature extractor for complementary image representations
3. **SVM**: Support Vector Machine trained on concatenated features from both CNNs

These base models are combined using a **Logistic Regression meta-learner** that learns optimal weights for prediction fusion.

## Features

- **Mixed precision training** for improved performance on modern GPUs
- **Data augmentation** (random flips, rotations, zoom) to prevent overfitting
- **Transfer learning** with ImageNet pre-trained weights
- **Fine-tuning** strategy for domain adaptation
- **Early stopping** and learning rate scheduling
- **Stacked ensemble** architecture for robust predictions

## Dataset

The model is trained on the DDR (Diabetic Retinopathy) dataset from Kaggle:
- **Dataset**: [mariaherrerot/ddrdataset](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset)
- **Classes**: Multiple severity levels (0-4)
- **Split**: 70% train, 20% validation, 10% test

## Requirements

```
tensorflow >= 2.x
numpy
pandas
scikit-learn
kagglehub
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DR_classification.git
cd DR_classification

# Install dependencies
pip install tensorflow numpy pandas scikit-learn kagglehub
```

## Usage

### Training the Model

```python
# The notebook automatically downloads the dataset
import kagglehub
path = kagglehub.dataset_download("mariaherrerot/ddrdataset")

# Run all cells in DR_class.ipynb
```

### Model Training Pipeline

1. **Data Loading & Preprocessing**
   - Load dataset from Kaggle
   - Standardize column names
   - Split into train/val/test sets (70/20/10)

2. **ResNet50 Training**
   - Initial training with frozen base (15 epochs)
   - Fine-tuning last 50 layers (5 epochs)
   - Learning rate: 1e-4 → 1e-6

3. **Feature Extraction**
   - Extract features from ResNet50 and VGG19
   - Concatenate features for SVM training

4. **SVM Training**
   - Train on stacked features
   - RBF kernel with C=10
   - Validation accuracy: **74.5%**

5. **Meta-Learner Ensemble**
   - Combine predictions from all models
   - Train logistic regression meta-learner
   - Final test accuracy: **75.8%**

## Model Performance

| Model | Validation Accuracy |
|-------|-------------------|
| ResNet50 (fine-tuned) | ~64.9% |
| SVM (stacked features) | 74.5% |
| **Ensemble (final)** | **75.8%** |

## Key Hyperparameters

- **Image Size**: 224×224
- **Batch Size**: 64
- **Initial Learning Rate**: 1e-4
- **Fine-tuning Learning Rate**: 1e-6
- **Weight Decay**: 1e-4
- **Label Smoothing**: 0.1
- **Dropout Rate**: 0.5

## Project Structure

```
DR_classification/
├── DR_class.ipynb              # Main training notebook
├── 75percentaccuracy_ensemblingStacking with VGG19.ipynb
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Techniques Used

- **Transfer Learning**: Leveraging ImageNet pre-trained weights
- **Fine-Tuning**: Unfreezing and retraining top layers
- **Data Augmentation**: Random flips, rotations, and zoom
- **Regularization**: L2 weight decay, dropout, label smoothing
- **Ensemble Learning**: Stacking multiple models with meta-learner
- **Mixed Precision**: FP16 training for faster computation

## Results Visualization

The model shows strong performance in detecting diabetic retinopathy, with the ensemble approach providing more robust predictions than individual models.

## Future Improvements

- [ ] Implement attention mechanisms
- [ ] Add class balancing techniques
- [ ] Experiment with EfficientNet architectures
- [ ] Add confusion matrix visualization
- [ ] Deploy as web application
- [ ] Add model interpretability (Grad-CAM)

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset: DDR Dataset by Maria Herrerot on Kaggle
- Pre-trained models: TensorFlow/Keras Applications
- Framework: TensorFlow 2.x

## Contact

For questions or collaboration opportunities, please open an issue or reach out via GitHub.

---

**Note**: This project was developed for educational and research purposes in medical image analysis.
