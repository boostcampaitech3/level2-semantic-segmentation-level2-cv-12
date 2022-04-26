import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.fcn_resnet50(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    def forward(self, x):
        return self.model(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_classes 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x
