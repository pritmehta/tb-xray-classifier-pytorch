# config.py
import torch

# --- Kaggle Dataset Configuration ---
KAGGLE_DATASET_ID = 'tawsifurrahman/tuberculosis-tb-chest-xray-dataset'
KAGGLE_DOWNLOAD_PATH = 'kaggle_dataset_downloaded'
RAW_DATA_SOURCE_DIR = f'{KAGGLE_DOWNLOAD_PATH}/TB_Chest_Radiography_Database'

# --- Processed Data Paths for PyTorch ---
DATA_PYTORCH_ROOT = 'data_pytorch'
TRAIN_DIR = f'{DATA_PYTORCH_ROOT}/train'
VAL_DIR = f'{DATA_PYTORCH_ROOT}/val'
TEST_DIR = f'{DATA_PYTORCH_ROOT}/test'

# --- Model & Training Parameters ---
IMG_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 20  # Epochs for initial head training
FINETUNE_EPOCHS = 15 # Epochs for fine-tuning the whole model
LEARNING_RATE = 0.0001
FINETUNE_LEARNING_RATE = 0.00001 # Smaller LR for fine-tuning
NUM_CLASSES = 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]

# Model saving
INITIAL_MODEL_SAVE_PATH = 'initial_best_tb_classifier_pytorch.pth' # For head training
FINAL_MODEL_SAVE_PATH = 'final_best_tb_classifier_pytorch.pth'     # For fine-tuned model

PATIENCE_EARLY_STOPPING = 7
FINETUNE_PATIENCE_EARLY_STOPPING = 5 # Potentially different patience for fine-tuning