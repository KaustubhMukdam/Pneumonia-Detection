# Technology Stack

## Initial choice

- Python 3.11+
- Jupyter notebooks for exploratory and experiment narratives
- TensorFlow/Keras for model training
- NumPy and pandas for data handling
- Matplotlib and seaborn for plots
- scikit-learn for metrics and evaluation utilities
- Pillow or TensorFlow image utilities for image inspection
- Git/GitHub for version control

## Execution environments

- Local Windows environment: documentation, Git, notebook editing, lightweight inspection.
- Kaggle Notebook GPU: repeatable training runs against the Kaggle dataset.
- Google Colab: fallback GPU environment only if Kaggle is unavailable.

## Selection rule

We will prefer the simplest tool that makes the experiment reproducible. New dependencies require a documented reason and an update to the eventual requirements file.
