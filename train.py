# train.py 6
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
from tqdm import tqdm

import config
import data_utils
import model_arch
from utils import plot_training_history

def train_model_pytorch(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, 
                        num_epochs, model_save_path, patience_early_stopping):
    start_time = time.time()
    
    best_model_wts_so_far = copy.deepcopy(model.state_dict()) # Track best weights for this training run
    best_val_loss_so_far = float('inf')
    epochs_without_improvement = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\nStarting training for {num_epochs} epochs on {config.DEVICE}...")
    print(f"Model will be saved to: {model_save_path}")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}", leave=False)
            for inputs, labels in progress_bar:
                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    preds_probs = torch.sigmoid(outputs)
                    preds_classes = (preds_probs > 0.5).float()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds_classes == labels.data)
                
                progress_bar.set_postfix(loss=loss.item(), acc=(torch.sum(preds_classes == labels.data).item()/inputs.size(0)))

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                if epoch_loss < best_val_loss_so_far:
                    print(f"  Validation loss decreased ({best_val_loss_so_far:.4f} --> {epoch_loss:.4f}). Saving model to {model_save_path}...")
                    best_val_loss_so_far = epoch_loss
                    best_model_wts_so_far = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_save_path)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(f"  Validation loss did not improve for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience_early_stopping:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break 

    time_elapsed = time.time() - start_time
    print(f'\nTraining phase complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Loss for this phase: {best_val_loss_so_far:4f}')

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path)) # Load best weights from this phase
    else: # Fallback if no model was saved (e.g., only 1 epoch, val loss didn't improve)
        model.load_state_dict(best_model_wts_so_far)

    return model, history


if __name__ == '__main__':
    print(f"Using PyTorch device: {config.DEVICE}")

    if not os.path.exists(config.TRAIN_DIR) or \
       (os.path.exists(config.TRAIN_DIR) and not os.listdir(os.path.join(config.TRAIN_DIR, 'Normal'))):
        print("Processed data directory for PyTorch not found or empty.")
        print("Please run `python download_data.py` and then `python data_utils.py` first.")
        exit()
    
    dataloaders, dataset_sizes, class_names, image_datasets = data_utils.get_dataloaders()
    if not dataloaders:
        print("Failed to get dataloaders. Exiting.")
        exit()

    # --- Initial Training Phase (Train Classifier Head) ---
    print("\n--- Starting Initial Training Phase (Classifier Head) ---")
    pytorch_model_initial = model_arch.get_pytorch_model(num_classes=config.NUM_CLASSES, pretrained=True)
    pytorch_model_initial = pytorch_model_initial.to(config.DEVICE)

    # Calculate pos_weight for BCEWithLogitsLoss
    train_targets = image_datasets['train'].targets
    count_class_0 = train_targets.count(0) 
    count_class_1 = train_targets.count(1) 

    if count_class_1 == 0:
        print("Warning: No samples of class 1 (Tuberculosis) found in training set. Defaulting pos_weight to 1.0.")
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(count_class_0) / count_class_1
    print(f"Training set class counts: Normal={count_class_0}, Tuberculosis={count_class_1}")
    print(f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight_value:.2f}")
    
    criterion_initial = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=config.DEVICE))
    optimizer_initial = optim.Adam(pytorch_model_initial.classifier[1].parameters(), lr=config.LEARNING_RATE)
    lr_scheduler_initial = lr_scheduler.StepLR(optimizer_initial, step_size=7, gamma=0.1)

    trained_model_initial, history_initial = train_model_pytorch(
        pytorch_model_initial, criterion_initial, optimizer_initial, lr_scheduler_initial, 
        dataloaders, dataset_sizes, num_epochs=config.INITIAL_EPOCHS,
        model_save_path=config.INITIAL_MODEL_SAVE_PATH,
        patience_early_stopping=config.PATIENCE_EARLY_STOPPING
    )
    
    print("\nInitial training phase finished.")
    if history_initial['train_loss']:
        print("Plotting initial training history...")
        plot_training_history(history_initial)

    # --- Fine-Tuning Phase (Train Entire Model or More Layers) ---
    print("\n--- Starting Fine-Tuning Phase ---")
    
    # Load the best model from the initial phase
    model_to_finetune = model_arch.get_pytorch_model(num_classes=config.NUM_CLASSES, pretrained=False) # Base structure
    if os.path.exists(config.INITIAL_MODEL_SAVE_PATH):
        model_to_finetune.load_state_dict(torch.load(config.INITIAL_MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"Loaded model from {config.INITIAL_MODEL_SAVE_PATH} for fine-tuning.")
    else:
        print(f"Warning: Initial model {config.INITIAL_MODEL_SAVE_PATH} not found. Fine-tuning might start from scratch or last state.")
        model_to_finetune.load_state_dict(trained_model_initial.state_dict()) # Use the model from previous phase in memory

    model_to_finetune = model_to_finetune.to(config.DEVICE)

    # Unfreeze all parameters for fine-tuning
    print("Unfreezing all model parameters for fine-tuning.")
    for param in model_to_finetune.parameters():
        param.requires_grad = True
    
    # Use the same criterion (with pos_weight) for fine-tuning
    criterion_finetune = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=config.DEVICE))
    
    optimizer_finetune = optim.Adam(
        filter(lambda p: p.requires_grad, model_to_finetune.parameters()), 
        lr=config.FINETUNE_LEARNING_RATE # Use a smaller learning rate
    )
    print(f"Fine-tuning optimizer with LR: {config.FINETUNE_LEARNING_RATE}")
    lr_scheduler_finetune = lr_scheduler.StepLR(optimizer_finetune, step_size=5, gamma=0.1)

    print(f"Fine-tuning for {config.FINETUNE_EPOCHS} epochs...")
    final_trained_model, history_finetune = train_model_pytorch(
        model_to_finetune, 
        criterion_finetune, 
        optimizer_finetune, 
        lr_scheduler_finetune, 
        dataloaders, 
        dataset_sizes, 
        num_epochs=config.FINETUNE_EPOCHS,
        model_save_path=config.FINAL_MODEL_SAVE_PATH, # Save to the final path
        patience_early_stopping=config.FINETUNE_PATIENCE_EARLY_STOPPING
    )

    print("\nFine-tuning phase finished.")
    if os.path.exists(config.FINAL_MODEL_SAVE_PATH):
        print(f"Best fine-tuned model saved to: {config.FINAL_MODEL_SAVE_PATH}")
    if history_finetune['train_loss']:
        print("Plotting fine-tuning history...")
        plot_training_history(history_finetune)

    print("\nFull training process complete.")