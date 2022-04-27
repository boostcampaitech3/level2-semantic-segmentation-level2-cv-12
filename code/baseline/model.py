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

class BaseModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.fcn_resnet50(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, 11, kernel_size=1)


    def forward(self, x):
        return self.model(x)

class FCNResnet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.fcn_resnet101(pretrained=True)
    
        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    def forward(self, x):
        return self.model(x)

class Deeplabv3_Resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(256, 11, kernel_size=1)

    def forward(self, x):
        return self.model(x)

class Deeplabv3_Resnet101(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)

        # output class를 data set에 맞도록 수정
        self.model.classifier[4] = nn.Conv2d(256, 11, kernel_size=1)
  
    def forward(self, x):
        return self.model(x)
class FCN8_Resnet101(nn.Module):
    def __init__(self, num_classes=12):
            super(FCN8_Resnet101, self).__init__()
            
            def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
                return nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                                out_channels=out_channels,
                                                kernel_size=kernel_size,
                                                stride=stride, 
                                                padding=padding),
                                    nn.ReLU(inplace=True)
                                    )        
            
            # conv1
            self.conv1_1 = CBR(3, 64, 3, 1, 1)
            self.conv1_2 = CBR(64, 64, 3, 1, 1)
            self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 
            
            # conv2
            self.conv2_1 = CBR(64, 128, 3, 1, 1)
            self.conv2_2 = CBR(128, 128, 3, 1, 1)  
            self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

            # conv3
            self.conv3_1 = CBR(128, 256, 3, 1, 1)
            self.conv3_2 = CBR(256, 256, 3, 1, 1)
            self.conv3_3 = CBR(256, 256, 3, 1, 1)          
            self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

            # Score pool3
            self.score_pool3_fr = nn.Conv2d(256,
                                            num_classes, 
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)             
            
            # conv4
            self.conv4_1 = CBR(256, 512, 3, 1, 1)
            self.conv4_2 = CBR(512, 512, 3, 1, 1)
            self.conv4_3 = CBR(512, 512, 3, 1, 1)          
            self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

            # Score pool4
            self.score_pool4_fr = nn.Conv2d(512,
                                            num_classes, 
                                            kernel_size=1,
                                            stride=1,
                                            padding=0)        
            
            # conv5
            self.conv5_1 = CBR(512, 512, 3, 1, 1)
            self.conv5_2 = CBR(512, 512, 3, 1, 1)
            self.conv5_3 = CBR(512, 512, 3, 1, 1)
            self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
            # fc6
            self.fc6 = nn.Conv2d(512, 4096, 1)
            self.relu6 = nn.ReLU(inplace=True)
            self.drop6 = nn.Dropout2d()

            # fc7
            self.fc7 = nn.Conv2d(4096, 4096, 1)
            self.relu7 = nn.ReLU(inplace=True)
            self.drop7 = nn.Dropout2d()


            # Score
            self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)
            
            
            # UpScore2 using deconv
            self.upscore2 = nn.ConvTranspose2d(num_classes,
                                            num_classes,
                                            kernel_size=4,
                                            stride=2,
                                            padding=1)
            
            # UpScore2_pool4 using deconv
            self.upscore2_pool4 = nn.ConvTranspose2d(num_classes, 
                                                    num_classes, 
                                                    kernel_size=4,
                                                    stride=2,
                                                    padding=1)
            
            # UpScore8 using deconv
            self.upscore8 = nn.ConvTranspose2d(num_classes, 
                                            num_classes,
                                            kernel_size=16,
                                            stride=8,
                                            padding=4)


    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.con3_3(h)        
        pool3 = h = self.pool3(h)
        # Score
        score_pool3c = self.score_pool3_fr(pool3)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)        
        pool4 = h = self.pool4(h)

        # Score
        score_pool4c = self.score_pool4_fr(pool4)       
        
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)        
        h = self.pool5(h)
        
        h = self.fc6(h)
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.drop7(h)
            
        h = self.score_fr(h)
            
        # Up Score I
        upscore2 = self.upscore2(h)
            
        # Sum I
        h = upscore2 + score_pool4c
            
        # Up Score II
        upscore2_pool4c = self.upscore2_pool4(h)
        
        # Sum II
        h = upscore2_pool4c + score_pool3c
            
        # Up Score III
        upscore8 = self.upscore8(h)
        
        return upscore8
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