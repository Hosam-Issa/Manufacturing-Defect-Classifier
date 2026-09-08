from torchvision import models
import torch.nn as nn

def build_model(device):
    # Load pretrained model
        model_ft = models.resnet18(weights='IMAGENET1K_V1')
        num_fts = model_ft.fc.in_features
        # Lower ouputs layer to 2 (ok vs. defective)
        model_ft.fc = nn.Linear(num_fts, 2)
        model_ft = model_ft.to(device)
        return model_ft