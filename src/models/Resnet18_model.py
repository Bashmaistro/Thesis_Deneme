import torch.nn as nn
import torchvision.models as models

def get_resnet18(num_classes=3):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model
