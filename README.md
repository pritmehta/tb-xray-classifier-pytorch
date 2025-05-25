<<<<<<< HEAD
# Tuberculosis (TB) X-ray Classifier - PyTorch Version

This project implements a Deep Learning model using PyTorch to classify chest X-ray images for Tuberculosis detection (Normal vs. Tuberculosis). It leverages transfer learning with a pre-trained MobileNetV2 architecture.

## Libraries Used

- **PyTorch & Torchvision:** For building and training the neural network.
- **NumPy:** For numerical operations.
- **Matplotlib & Seaborn:** For plotting training history and confusion matrices.
- **scikit-learn:** For performance metrics (classification report, accuracy, confusion matrix) and data splitting.
- **Kaggle API:** For downloading the dataset directly from Kaggle.
- **OpenCV-Python:** Listed as available; can be used for advanced image processing (though `torchvision.transforms` and `Pillow` are primarily used here for image handling).
- **Pillow (PIL):** For image loading.
- **tqdm:** For progress bars during training and evaluation.
- **TensorFlow:** Listed as an available library (not used in this core PyTorch pipeline).

## Project Structure

tuberculosis_classifier_pytorch/
├── kaggle_dataset_downloaded/ # Where Kaggle API will download the raw dataset
│ └── tuberculosis-tb-chest-xray-dataset/
│ └── TB_Chest_Radiography_Database/
│ ├── Normal/
│ └── Tuberculosis/
├── data_pytorch/ # Processed data for PyTorch (train/val/test)
│ ├── train/
│ │ ├── Normal/
│ │ └── Tuberculosis/
│ ├── val/
│ # ... and so on
│ └── test/
│ # ... and so on
├── config.py # All configurations
├── download_data.py # Script to download data via Kaggle API
├── data_utils.py # Data splitting, transforms, dataloaders
├── model_arch.py # Neural network architecture
├── utils.py # Utility functions (e.g., plotting)
├── train.py # Training script
├── evaluate.py # Evaluation script
├── predict.py # Prediction on new images
└── README.md # Project instructions

## Setup Instructions

1.  **Clone the Repository (if applicable) or Create Project Directory:**
    Set up your project folder.

2.  **Install Dependencies:**

    ```bash
    pip install torch torchvision torchaudio numpy scikit-learn Pillow matplotlib seaborn opencv-python kaggle tqdm tensorflow
    ```

3.  **Set up Kaggle API Token:**
    - Go to your Kaggle account page: `https://www.kaggle.com/YOUR_USERNAME/account`.
    - Click "Create New API Token" to download `kaggle.json`.
    - Place `kaggle.json` in the correct directory:
      - Linux/macOS: `~/.kaggle/kaggle.json`
      - Windows: `C:\Users\YOUR_WINDOWS_USERNAME\.kaggle\kaggle.json`
    - Set file permissions (e.g., `chmod 600 ~/.kaggle/kaggle.json` on Linux/macOS).

## Workflow

1.  **Download Dataset from Kaggle:**
    This script uses the Kaggle API to download and extract the dataset into the `kaggle_dataset_downloaded/` folder.

    ```bash
    python download_data.py
    ```

2.  **Prepare and Split Data for PyTorch:**
    This script takes the raw downloaded data, splits it into training, validation, and test sets, and organizes it into the `data_pytorch/` directory. Run this after downloading.

    ```bash
    python data_utils.py
    ```

3.  **Train the Neural Network:**
    This script trains the PyTorch model using the data in `data_pytorch/`. The best model weights (based on validation loss) will be saved to `best_tb_classifier_pytorch.pth`. Training history plots will be displayed.

    ```bash
    python train.py
    ```

4.  **Evaluate the Trained Model:**
    Load the saved model and evaluate its performance on the test set. This will print a classification report and display a confusion matrix.

    ```bash
    python evaluate.py
    ```

5.  **Predict on a New X-ray Image:**
    Use the trained model to classify a new, unseen X-ray image.
    ```bash
    python predict.py /path/to/your/xray_image.png
    ```
    Replace `/path/to/your/xray_image.png` with the actual path to an image file.

## Configuration

All major parameters (paths, image size, batch size, epochs, learning rate, etc.) can be configured in `config.py`.

## Notes

- The model uses a MobileNetV2 base, pre-trained on ImageNet. Only the final classifier layer is trained initially. For advanced fine-tuning, you can modify `train.py` to unfreeze more layers of the base model and use a smaller learning rate.
- Early stopping is implemented to prevent overfitting by monitoring validation loss.
- The `TensorFlow` library is listed as an installation but is not actively used in this PyTorch-centric neural network pipeline. It's included as per the user's specified library list.
- `OpenCV-Python` is available for more complex image manipulations if needed, though `torchvision.transforms` and `Pillow` handle the current image processing tasks.
=======
# tb-xray-classifier-pytorch
A neural network which detects tuberculosis from chest x-ray images
>>>>>>> ac0d9cc940bad04f81328aef3d729aef16bf2c7e
