# Tuberculosis (TB) X-ray Classifier – PyTorch

A deep learning project that classifies chest X-ray images as **Normal** or **Tuberculosis**
using transfer learning with a pre-trained **MobileNetV2** model in PyTorch.

## Tech Stack
- PyTorch, Torchvision
- NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Kaggle API
- Pillow, tqdm

## Features
- End-to-end deep learning pipeline
- Transfer learning using MobileNetV2
- Early stopping to prevent overfitting
- Model evaluation using confusion matrix and classification report

## Project Structure
```text
tuberculosis_classifier_pytorch/
├── download_data.py
├── data_utils.py
├── train.py
├── evaluate.py
├── predict.py
├── model_arch.py
├── config.py
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python download_data.py
python data_utils.py
python train.py
python evaluate.py
```

## Dataset

TB Chest Radiography Dataset (via Kaggle)






