# predict.py    8
# predict.py
import torch
from PIL import Image
import os
import argparse

import config
import data_utils
import model_arch

def predict_tb_from_image(image_path, pytorch_model, class_names_list, device=config.DEVICE):
    pytorch_model.eval()
    image_transform = data_utils.get_data_transforms()['test'] 
    
    try:
        image = Image.open(image_path).convert('RGB') 
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None, None
    except Exception as e:
        print(f"Error opening or processing image '{image_path}': {e}")
        return None, None

    image_tensor = image_transform(image).unsqueeze(0) 
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output_logits = pytorch_model(image_tensor)
        probabilities = torch.sigmoid(output_logits)
        prob_class_1 = probabilities.item()

    if prob_class_1 > 0.5:
        predicted_label_str = class_names_list[1] 
        confidence_score = prob_class_1
    else:
        predicted_label_str = class_names_list[0]
        confidence_score = 1 - prob_class_1
        
    print(f"\n--- Prediction for Image: {os.path.basename(image_path)} ---")
    print(f"  Raw Logit Output: {output_logits.item():.4f}")
    print(f"  Probability of being '{class_names_list[1]}': {prob_class_1:.4f}")
    print(f"  Predicted Label: {predicted_label_str}")
    print(f"  Confidence in Prediction: {confidence_score*100:.2f}%")
    
    return predicted_label_str, confidence_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Tuberculosis from an X-ray image using a trained PyTorch model.")
    parser.add_argument("image_path", type=str, help="Path to the X-ray image file for prediction.")
    args = parser.parse_args()

    print(f"Using PyTorch device: {config.DEVICE}")

    pytorch_model_loaded = model_arch.get_pytorch_model(num_classes=config.NUM_CLASSES, pretrained=False)
    
    # Ensure you are loading the FINAL fine-tuned model
    if not os.path.exists(config.FINAL_MODEL_SAVE_PATH):
        print(f"Error: Trained model file not found at '{config.FINAL_MODEL_SAVE_PATH}'.")
        print(f"Looking for initial model '{config.INITIAL_MODEL_SAVE_PATH}' as fallback for prediction (not recommended).")
        if not os.path.exists(config.INITIAL_MODEL_SAVE_PATH):
            print(f"Error: Neither final nor initial model found. Please train the model first by running `python train.py`.")
            exit()
        else:
            model_load_path = config.INITIAL_MODEL_SAVE_PATH
    else:
        model_load_path = config.FINAL_MODEL_SAVE_PATH
        
    try:
        pytorch_model_loaded.load_state_dict(torch.load(model_load_path, map_location=config.DEVICE))
        print(f"Successfully loaded trained model from '{model_load_path}'.")
    except Exception as e:
        print(f"Error loading model state_dict from '{model_load_path}': {e}")
        exit()
    pytorch_model_loaded = pytorch_model_loaded.to(config.DEVICE)

    class_names_inferred = None
    try:
        if os.path.exists(config.TRAIN_DIR):
            dummy_train_dataset = data_utils.datasets.ImageFolder(config.TRAIN_DIR)
            class_names_inferred = dummy_train_dataset.classes
            print(f"Inferred class names from dataset: {class_names_inferred} (0: {class_names_inferred[0]}, 1: {class_names_inferred[1]})")
        else:
            raise FileNotFoundError (f"Train directory {config.TRAIN_DIR} not found for inferring class names.")
    except Exception as e:
        print(f"Warning: Could not automatically infer class names. Error: {e}")
        print("Assuming default: ['Normal', 'Tuberculosis']")
        class_names_inferred = ['Normal', 'Tuberculosis']

    predicted_class, confidence = predict_tb_from_image(args.image_path, pytorch_model_loaded, class_names_inferred)

    if predicted_class:
        print(f"Final Prediction: The image is classified as '{predicted_class}' with {confidence*100:.2f}% confidence.")
    else:
        print("Prediction failed. Please check the image path and model.")