import torch
import torch.nn as nn
import os
import torchvision.models as models

from torchvision.models import resnet18, ResNet18_Weights

def resnet18_revise_model(pretrained=True):
    if pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)


    return model