# data_utils.py-3
import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch
# import cv2

import config

def split_dataset_files(
    source_dir=config.RAW_DATA_SOURCE_DIR,
    train_dir=config.TRAIN_DIR,
    val_dir=config.VAL_DIR,
    test_dir=config.TEST_DIR,
    train_size=0.7,
    val_size=0.15
):
    print("Starting dataset split for PyTorch...")
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found. Please run download_data.py first.")
        return

    for dir_path in [train_dir, val_dir, test_dir]:
        for class_name in ['Normal', 'Tuberculosis']:
            os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)

    for class_name in ['Normal', 'Tuberculosis']:
        class_source_path = os.path.join(source_dir, class_name)
        if not os.path.exists(class_source_path):
            print(f"Warning: Source class directory '{class_source_path}' not found. Skipping.")
            continue

        files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
        
        if not files:
            print(f"No files found in {class_source_path}. Skipping split for this class.")
            continue

        train_files, temp_files = train_test_split(files, train_size=train_size, random_state=42, shuffle=True)
        
        if not temp_files:
            val_files, test_files = [], []
        else:
            relative_val_size = val_size / (1 - train_size) if (1 - train_size) > 0 else 0
            if relative_val_size >= 1.0 and len(temp_files) > 0:
                 val_files = temp_files
                 test_files = []
            elif relative_val_size <= 0.0:
                 val_files = []
                 test_files = temp_files
            else:
                val_files, test_files = train_test_split(temp_files, train_size=relative_val_size, random_state=42, shuffle=True)

        def copy_files_to_dest(file_list, dest_folder_root):
            dest_class_folder = os.path.join(dest_folder_root, class_name)
            for f_name in file_list:
                shutil.copy(os.path.join(class_source_path, f_name),
                            os.path.join(dest_class_folder, f_name))

        copy_files_to_dest(train_files, train_dir)
        copy_files_to_dest(val_files, val_dir)
        copy_files_to_dest(test_files, test_dir)
        
        print(f"Class: {class_name}")
        print(f"  Total original: {len(files)}")
        print(f"  Train: {len(train_files)}")
        print(f"  Val:   {len(val_files)}")
        print(f"  Test:  {len(test_files)}")
    print("Dataset split complete.")


def get_data_transforms():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.85, 1.0)), # Slightly less aggressive crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10), # Reduced rotation
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=3), # Gentle affine
            # transforms.ColorJitter(brightness=0.1, contrast=0.1), # Subtle color jitter if needed
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN_IMAGENET, config.STD_IMAGENET)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN_IMAGENET, config.STD_IMAGENET)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN_IMAGENET, config.STD_IMAGENET)
        ]),
    }
    return data_transforms

def get_dataloaders():
    data_transforms_dict = get_data_transforms()

    if not os.path.exists(config.TRAIN_DIR):
        print(f"Error: Processed data directory '{config.TRAIN_DIR}' not found. Run data splitting first.")
        return None, None, None, None

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(config.DATA_PYTORCH_ROOT, x), data_transforms_dict[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], 
            batch_size=config.BATCH_SIZE, 
            shuffle=(x=='train'),
            num_workers=min(os.cpu_count(), 4)
        )
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print("DataLoaders Information:")
    print("  Class names:", class_names, " (0:", class_names[0], ", 1:", class_names[1], ")")
    print("  Dataset sizes:", dataset_sizes)
    
    return dataloaders, dataset_sizes, class_names, image_datasets # Return image_datasets for pos_weight calc

if __name__ == '__main__':
    if not os.path.exists(config.RAW_DATA_SOURCE_DIR):
        print(f"Raw data source '{config.RAW_DATA_SOURCE_DIR}' not found.")
        print("Please run `python download_data.py` first to download and extract the dataset.")
    else:
        if not os.path.exists(config.TRAIN_DIR) or \
           (os.path.exists(config.TRAIN_DIR) and not os.listdir(os.path.join(config.TRAIN_DIR, 'Normal'))):
            print("Processed data directory for PyTorch not found or empty. Proceeding with data split.")
            split_dataset_files()
        else:
            print("Data already split for PyTorch. To re-split, delete the "
                  f"'{config.DATA_PYTORCH_ROOT}' directory first.")
        
        loaders, sizes, names, img_datasets = get_dataloaders()
        if loaders:
            print(f"\n  Train Dataloader: {len(loaders['train'])} batches")
            print(f"  Validation Dataloader: {len(loaders['val'])} batches")
            print(f"  Test Dataloader: {len(loaders['test'])} batches")