import torch.nn as nn
from torchvision import models


class Fcn_Resnet50:
    def __init__(self):
        
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
        
        self.model = model
        
    def __call__(self):
        return self.model