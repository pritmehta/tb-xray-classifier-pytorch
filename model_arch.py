# model_arch.py - 4
import torch
import torch.nn as nn
from torchvision import models
import config

def get_pytorch_model(num_classes=config.NUM_CLASSES, pretrained=True):
    """
    Loads a pre-trained MobileNetV2 model and replaces its classifier
    for the specified number of classes.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Freeze all base model parameters if using pretrained weights
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False # Parameters are frozen by default

    # Get the number of input features for the classifier part of MobileNetV2
    # MobileNetV2's classifier is a Sequential(Linear, Dropout, Linear) or just Linear for some versions.
    # We usually target the last Linear layer. For mobilenet_v2, it's model.classifier[1]
    num_ftrs = model.classifier[1].in_features

    # Replace the classifier head
    # For binary classification with BCEWithLogitsLoss, we need 1 output neuron.
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    print("PyTorch Model Architecture (MobileNetV2 base with modified classifier):")
    print("  Classifier head:", model.classifier)
    
    return model

if __name__ == '__main__':
    # Example of how to get the model
    model = get_pytorch_model()
    print("\nExample PyTorch model loaded successfully.")
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE) # (batch, channels, height, width)
    output = model(dummy_input)
    print("  Dummy input shape:", dummy_input.shape)
    print("  Dummy output shape (logits):", output.shape) # Should be [1, 1] for binary