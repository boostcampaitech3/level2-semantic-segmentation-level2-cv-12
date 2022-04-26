import torch.nn as nn
import torch.optim as optim
from torchvision import models

class FCNResNet50(nn.Module):
    def __init__(self, num_classes):
        super(FCNResNet50, self).__init__()
        self.model = models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    def forward(self, x):
        x = self.model(x)
        return x
