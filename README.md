# Automated Pneumonia Detection from Chest X-Rays using Deep Learning

*A Comparative Study of CNN Architectures for Medical Image Classification*

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This project implements and compares three deep learning models for automated pneumonia detection from chest X-ray images. The models include a simple CNN baseline, a regularized CNN with dropout, and a transfer learning approach using VGG16. The VGG16 model achieved the highest test accuracy of 85.90%.

The project demonstrates the application of computer vision techniques in medical imaging, providing insights into model architectures suitable for diagnostic tasks.

## Dataset

**Source:** [Kaggle - Chest X-Ray Images (Paul Mooney)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Composition:**
- Total Images: 5,863 X-ray images in JPEG format
- Organization: Three folders (Train, Test, Val) with two subfolders per category (Pneumonia/Normal)
- Image Type: Grayscale chest X-rays

**Preprocessing:**
- Normalization: Pixel values rescaled to [0, 1]
- Data Augmentation: Random rotation, zoom, shearing, and horizontal flip for training data
- Image Resizing: All images standardized to 150 × 150 pixels

## Prerequisites

- Python 3.11+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install required packages:
   ```bash
   pip install tensorflow keras numpy matplotlib scikit-learn jupyter
   ```

3. Download the dataset from Kaggle and place it in the project directory as `chest_xray/` folder.

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook x_ray.ipynb
   ```

2. Update the `BASE_DIR` path in the configuration section if needed:
   ```python
   BASE_DIR = './chest_xray/'  # Update this path to where you unzipped the dataset
   ```

3. Run the notebook cells sequentially to:
   - Load and preprocess the data
   - Train the three models
   - Evaluate performance on the test set

## Model Architectures

### Model 1: Simple CNN (Baseline)
- Two convolutional blocks with max-pooling
- Dense layers for classification
- **Test Accuracy:** 82.37%

### Model 2: CNN with Dropout
- Three convolutional blocks with max-pooling
- Dropout layer (rate: 0.5) for regularization
- **Test Accuracy:** 80.77%

### Model 3: VGG16 Transfer Learning
- Pre-trained VGG16 base (frozen weights)
- Custom dense classifier on top
- **Test Accuracy:** 85.90% (Best performer)

## Results Summary

| Model | Architecture Type | Test Accuracy | Test Loss |
|-------|------------------|---------------|-----------|
| Model 1 | Simple CNN | 82.37% | 0.3949 |
| Model 2 | CNN + Dropout | 80.77% | 0.4488 |
| Model 3 | VGG16 Transfer | **85.90%** | 0.3970 |

The VGG16 transfer learning model outperformed the custom architectures, demonstrating the effectiveness of leveraging pre-trained features for medical imaging tasks with limited labeled data.

## Project Structure

```
├── README.md                 # Project documentation
├── report.html              # Detailed HTML report
├── report.md                # Markdown version of the report
├── x_ray.ipynb             # Main Jupyter notebook with code
├── chest_xray/             # Dataset directory
│   ├── train/
│   ├── test/
│   └── val/
├── image_1.png             # Training curves for Model 1
├── image_2.png             # Training curves for Model 2
├── image_3.png             # Training curves for Model 3
└── .gitignore              # ignore dataset and extra files
```

## Key Findings

- **Transfer Learning Superiority:** Pre-trained models significantly outperform custom architectures for medical imaging
- **Regularization Trade-offs:** Dropout helps prevent overfitting but may require more training epochs
- **Clinical Potential:** 85.90% accuracy shows promise for automated screening, though human oversight remains essential

For detailed analysis, training dynamics, and future recommendations, see the [full report](report.html).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```
Automated Pneumonia Detection from Chest X-Rays using Deep Learning
A Comparative Study of CNN Architectures for Medical Image Classification
```

## Contact

For questions or feedback, please open an issue in this repository.
