# utils.py  - 5
import matplotlib.pyplot as plt
import numpy as np # NumPy is used here
import seaborn as sns
from sklearn.metrics import confusion_matrix # scikit-learn is used here
import torch

def plot_training_history(history):
    """Plots training and validation accuracy and loss from the history dictionary."""
    # Ensure values are NumPy arrays or Python scalars for plotting
    acc = [h.cpu().numpy() if torch.is_tensor(h) else h for h in history['train_acc']]
    val_acc = [h.cpu().numpy() if torch.is_tensor(h) else h for h in history['val_acc']]
    loss = history['train_loss']
    val_loss = history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6)) # Matplotlib is used here

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o', linestyle='-')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='x', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='x', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names):
    """Plots the confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5,
                xticklabels=class_names, yticklabels=class_names) # Seaborn is used here
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show()