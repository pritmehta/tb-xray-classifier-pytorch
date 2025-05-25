# evaluate.py   -7
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import os

import config
import data_utils
import model_arch
from utils import plot_confusion_matrix_heatmap

def evaluate_pytorch_model(model, dataloader, criterion):
    model.eval()
    all_predicted_labels = []
    all_true_labels = []
    running_loss = 0.0
    running_corrects = 0
    dataset_size = len(dataloader.dataset)

    print(f"\nEvaluating model on {config.DEVICE}...")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, labels in progress_bar:
            inputs = inputs.to(config.DEVICE)
            true_labels_batch = labels.to(config.DEVICE).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, true_labels_batch)
            
            preds_probs = torch.sigmoid(outputs)
            predicted_classes_batch = (preds_probs > 0.5).float()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predicted_classes_batch == true_labels_batch.data)

            all_predicted_labels.extend(predicted_classes_batch.cpu().numpy().flatten())
            all_true_labels.extend(labels.cpu().numpy().flatten())
            
            progress_bar.set_postfix(loss=loss.item())

    test_loss = running_loss / dataset_size
    test_acc_sklearn = accuracy_score(all_true_labels, all_predicted_labels)
    
    print(f'\nTest Set Evaluation Results:')
    print(f'  Loss: {test_loss:.4f}')
    print(f'  Accuracy (sklearn): {test_acc_sklearn:.4f}')
    
    return np.array(all_true_labels), np.array(all_predicted_labels)


if __name__ == '__main__':
    print(f"Using PyTorch device: {config.DEVICE}")

    dataloaders, _, class_names, _ = data_utils.get_dataloaders() # Don't need dataset_sizes or image_datasets here
    if not dataloaders or 'test' not in dataloaders:
        print("Failed to get test dataloader. Ensure data is processed and paths are correct.")
        exit()
    test_dataloader = dataloaders['test']
    
    if not test_dataloader.dataset or len(test_dataloader.dataset) == 0 :
        print("Test dataset is empty. Please check your data splitting and paths.")
        exit()

    pytorch_model = model_arch.get_pytorch_model(num_classes=config.NUM_CLASSES, pretrained=False)
    
    # Ensure you are loading the FINAL fine-tuned model
    if not os.path.exists(config.FINAL_MODEL_SAVE_PATH):
        print(f"Error: Trained model file not found at '{config.FINAL_MODEL_SAVE_PATH}'.")
        print(f"Looking for initial model '{config.INITIAL_MODEL_SAVE_PATH}' as fallback for evaluation (not recommended for final eval).")
        if not os.path.exists(config.INITIAL_MODEL_SAVE_PATH):
            print(f"Error: Neither final nor initial model found. Please train the model first by running `python train.py`.")
            exit()
        else:
            model_load_path = config.INITIAL_MODEL_SAVE_PATH
    else:
        model_load_path = config.FINAL_MODEL_SAVE_PATH
        
    try:
        pytorch_model.load_state_dict(torch.load(model_load_path, map_location=config.DEVICE))
        print(f"Loaded trained model from '{model_load_path}'.")
    except Exception as e:
        print(f"Error loading model state_dict from '{model_load_path}': {e}")
        exit()
        
    pytorch_model = pytorch_model.to(config.DEVICE)
    
    criterion = nn.BCEWithLogitsLoss() # Use a basic criterion for eval loss calculation

    true_labels, predicted_labels = evaluate_pytorch_model(pytorch_model, test_dataloader, criterion)

    print("\nScikit-learn Classification Report:")
    if class_names:
        print(classification_report(true_labels, predicted_labels, target_names=class_names, digits=4))
        plot_confusion_matrix_heatmap(true_labels, predicted_labels, class_names)
    else:
        print("Class names not available for report. Raw report:")
        print(classification_report(true_labels, predicted_labels, digits=4))