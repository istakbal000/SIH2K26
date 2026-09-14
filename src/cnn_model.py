import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torchvision import models

def get_fire_cnn_model(pretrained=True):
    """
    Returns a ResNet18 model modified for binary classification (Fire vs No-Fire).
    """
    # Load a pretrained ResNet18 model
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Freeze earlier layers to speed up training and prevent overfitting on small datasets
    # We will only train the last residual block and the fully connected layer
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
            
    # Modify the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 1) # Binary output (logit)
    )
    
    return model
