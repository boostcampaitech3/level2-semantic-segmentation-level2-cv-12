from tkinter import N
from turtle import forward
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import numpy as np
"""
사용가능 모델 
사용방법 --model 모델명
BaseModel,
BaseModel2,
FCNResnet101,
Deeplabv3_Resnet50,
Deeplabv3_Resnet101,
UnetPlusPlus_Resnet50,
DeconvNet_VGG16,
FCN8_VGG16,
UNet,
UnetPlusPlus_Efficient4,
UnetPlusPlus_Efficient_b5,

"""

from models import STLNet


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

# use smp
class UNet_Resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "resnet50"      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.Unet(         # choose architecture 
            encoder_name=encoder_name,     
            encoder_weights="imagenet",   
            in_channels=3,              
            classes=11,                 
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UNet_Resnet101(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "resnet101"  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.Unet(      # choose architecture 
            encoder_name=encoder_name,         
            encoder_weights="imagenet",    
            in_channels=3,                  
            classes=11,                      
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UnetPlusPlus_Resnet50(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "resnet50"        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(   # choose architecture 
            encoder_name=encoder_name,     
            encoder_weights="imagenet", 
            in_channels=3,              
            classes=11,                
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UnetPlusPlus_Efficient4(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "efficientnet-b4"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="imagenet",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)
    
class UnetPlusPlus_Efficient7(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "efficientnet-b7"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="imagenet",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)    

class UnetPlusPlus_Efficient5_N(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "timm-efficientnet-b5"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="noisy-student",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='noisy-student')
    
    def forward(self, x):
        return self.model(x)    

class UnetPlusPlus_Efficient_b5(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "efficientnet-b5"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="imagenet",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UnetPlusPlus_Efficient_b7(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_name = "efficientnet-b7"   # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(     # choose architecture
            encoder_name=encoder_name, 
            encoder_weights="imagenet",     
            in_channels=3,       
            classes=11,              
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)
        
class Deeplabv3Plus_Resnet101(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "resnet101"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.DeepLabV3Plus(    # choose architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class Deeplabv3Plus_Resnext50(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "resnext50_32x4d"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.DeepLabV3Plus(    # choose architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class Deeplabv3Plus_Resnext101(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "resnext101_32x8d"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.DeepLabV3Plus(    # choose architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class Deeplabv3Plus_SEResnet152(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "se_resnet152"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.DeepLabV3Plus(    # choose architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class UNetPlusPlus_HRNet30(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_name = "tu-hrnet_w30"         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        self.model = smp.UnetPlusPlus(    # choosqe architecture
            encoder_name=encoder_name,  
            encoder_weights="imagenet",    
            in_channels=3,                
            classes=11,                    
        )

        self.preprocess_input = get_preprocessing_fn(encoder_name, pretrained='imagenet')
    
    def forward(self, x):
        return self.model(x)

class DeconvNet_VGG16(nn.Module):
    def __init__(self, num_classes=11):
        super(DeconvNet_VGG16, self).__init__()

        def Conv_Block(in_channels,out_channels,kernel_size,stride,padding):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        def Deconv_Block(in_channels,out_channels,kernel_size,stride,padding):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        self.conv1_1 = Conv_Block(3,64,3,1,1)
        self.conv1_2 = Conv_Block(64,64,3,1,1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        self.conv2_1 = Conv_Block(64,128,3,1,1)
        self.conv2_2 = Conv_Block(128,128,3,1,1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        self.conv3_1 = Conv_Block(128,256,3,1,1)
        self.conv3_2 = Conv_Block(256,256,3,1,1)
        self.conv3_3 = Conv_Block(256,256,3,1,1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)
        
        self.conv4_1 = Conv_Block(256,512,3,1,1)
        self.conv4_2 = Conv_Block(512,512,3,1,1)
        self.conv4_3 = Conv_Block(512,512,3,1,1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        self.conv5_1 = Conv_Block(512,512,3,1,1)        
        self.conv5_2 = Conv_Block(512,512,3,1,1)
        self.conv5_3 = Conv_Block(512,512,3,1,1)        
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        self.fc6= Conv_Block(512,4096,7,1,0)
        self.drop6 = nn.Dropout2d(0.5)
        self.fc7= nn.Conv2d(4096,4096,1,1,0)
        self.drop7 = nn.Dropout2d(0.5)

        self.fc6_deconv = Deconv_Block(4096,512,7,1,0)
        
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv5_1 = Deconv_Block(512,512,3,1,1) 
        self.deconv5_2 = Deconv_Block(512,512,3,1,1)
        self.deconv5_3 = Deconv_Block(512,512,3,1,1)

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv4_1  = Deconv_Block(512,512,3,1,1) 
        self.deconv4_2  = Deconv_Block(512,512,3,1,1)
        self.deconv4_3  = Deconv_Block(512,256,3,1,1)

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv3_1 = Deconv_Block(256,256,3,1,1)
        self.deconv3_2 = Deconv_Block(256,256,3,1,1)
        self.deconv3_3 = Deconv_Block(256,128,3,1,1)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv2_1 = Deconv_Block(128,128,3,1,1)
        self.deconv2_2 = Deconv_Block(128,64,3,1,1)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv1_1 = Deconv_Block(64,64,3,1,1)
        self.deconv1_2 = Deconv_Block(64,64,3,1,1)

        self.score_fr = nn.Conv2d(64,num_classes,1,1,0,1)

    def forward(self, x):
        
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h,pool1_indices = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h,pool2_indices = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h,pool3_indices = self.pool3(h)
        
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        h,pool4_indices = self.pool4(h)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h,pool5_indices = self.pool5(h)

        h = self.fc6(h)
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.drop7(h)

        h = self.fc6_deconv(h)

        h = self.unpool5(h,pool5_indices)
        h = self.deconv5_1(h)
        h = self.deconv5_2(h)
        h = self.deconv5_3(h)

        h = self.unpool4(h,pool4_indices)
        h = self.deconv4_1(h)
        h = self.deconv4_2(h)
        h = self.deconv4_3(h)

        
        h = self.unpool3(h,pool3_indices)
        h = self.deconv3_1(h)
        h = self.deconv3_2(h)
        h = self.deconv3_3(h)

        h = self.unpool2(h,pool2_indices)
        h = self.deconv2_1(h)
        h = self.deconv2_2(h)

        h = self.unpool1(h,pool1_indices)
        h = self.deconv1_1(h)
        h = self.deconv1_2(h)
        
        return self.score_fr(h)

class FCN8_VGG16(nn.Module):
    def __init__(self, num_classes=12):
        super(FCN8_VGG16, self).__init__()
        
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
        h = self.conv3_3(h)        
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
    
class UNet(nn.Module):
    def __init__(self, num_classes=11):
        super(UNet, self).__init__()
        def encoder(in_channels,out_channels,kernel_size,stride,padding,bias):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=bias
                        ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        def decoder(in_channels,out_channels,kernel_size,stride,padding,bias):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
            )

        self.conv1_1 = encoder(3,64,3,1,1,True)
        self.conv1_2 = encoder(64,64,3,1,1,True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = encoder(64,128,3,1,1,True)
        self.conv2_2 = encoder(128,128,3,1,1,True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = encoder(128,256,3,1,1,True)
        self.conv3_2 = encoder(256,256,3,1,1,True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = encoder(256,512,3,1,1,True)
        self.conv4_2 = encoder(512,512,3,1,1,True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = encoder(512,1024,3,1,1,True)
        self.conv5_2 = encoder(1024,1024,3,1,1,True)

        self.unpool4 = decoder(1024,512,2,2,0,True)
        self.deconv4_2 = encoder(1024,512,3,1,1,True)
        self.deconv4_1 = encoder(512,512,3,1,1,True)
        
        self.unpool3 = decoder(512,256,2,2,0,True)
        self.deconv3_2 = encoder(512,256,3,1,1,True)
        self.deconv3_1 = encoder(256,256,3,1,1,True)

        self.unpool2 = decoder(256,128,2,2,0,True)
        self.deconv2_2 = encoder(256,128,3,1,1,True)
        self.deconv2_1 = encoder(128,128,3,1,1,True)

        self.unpool1 = decoder(128,64,2,2,0,True)
        self.deconv1_2 = encoder(128,64,3,1,1,True)
        self.deconv1_1 = encoder(64,64,3,1,1,True)
        #output samentic map
        self.score_fr = nn.Conv2d(in_channels=64,out_channels=num_classes,kernel_size=1,stride=1,padding=0,bias=True)
        
    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        
        unpool4 = self.unpool4(conv5_2)
        
        upsample4 = torch.cat((unpool4,conv4_2),dim=1)
        
        deconv4_2 = self.deconv4_2(upsample4)
        deconv4_1 = self.deconv4_1(deconv4_2)

        unpool3 = self.unpool3(deconv4_1)
        upsample3 = torch.cat((unpool3,conv3_2),dim=1)
        deconv3_2 = self.deconv3_2(upsample3)
        deconv3_1 = self.deconv3_1(deconv3_2)

        unpool2 = self.unpool2(deconv3_1)
        upsample2 = torch.cat((unpool2,conv2_2),dim=1)
        deconv2_2 = self.deconv2_2(upsample2)
        deconv2_1 = self.deconv2_1(deconv2_2)

        unpool1 = self.unpool1(deconv2_1)
        upsample1 = torch.cat((unpool1,conv1_2),dim=1)
        deconv1_2 = self.deconv1_2(upsample1)
        deconv1_1 = self.deconv1_1(deconv1_2)
        
        return  self.score_fr(deconv1_1) 

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch,out_ch,k_size,stride,padding,dilation=1,relu=True):
    block = []
    block.append(nn.Conv2d(in_ch,out_ch,k_size,stride,padding,dilation,bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)

class Bottleneck(nn.Module):
    def __init__(self,in_ch,out_ch,stride,dilation=1,downsample=False):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(in_ch,out_ch//4, 1, 1, 0),
            conv_block(out_ch//4,out_ch//4, 3 ,stride,dilation,dilation),
            conv_block(out_ch//4,out_ch, 1 ,1 ,0,relu=False)
        )
        self.downsample = nn.Sequential(
            conv_block(in_ch,out_ch, 1, stride, 0, 1, False),
        ) if downsample else None
        self.relu = nn.ReLU()
    def forward(self,x):
        identity = x
        out = self.block(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride,dilation,num_layers):
        super().__init__()
        block = []
        for i in range(num_layers):
            block.append(Bottleneck(
                in_ch if i==0 else out_ch,
                out_ch,
                stride if i==0 else 1,
                dilation,
                True if i==0 else False
            ))
        self.block = nn.Sequential(*block)
    def forward(self,x):
        return self.block(x)

class ResNet101(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(in_channels,64,7,2,3),
            nn.MaxPool2d(3,2,1),
            ResBlock(64,256,1,1,num_layers=3),
            ResBlock(256,512,2,1,num_layers=4),
            ResBlock(512,1024,1,2,num_layers=23),
            ResBlock(1024,2048,1,4,num_layers=3)
        )
    def forward(self,x):
        return self.block(x)

class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        #1x1 conv
        self.block1 = conv_block(in_channels,out_channels,1,1,0)
        #3x3 conv rate=6
        self.block2 = conv_block(in_channels,out_channels,3,1,6,dilation=6)
        #3x3 conv rate=12
        self.block3 = conv_block(in_channels,out_channels,3,1,12,dilation=12)
        #3x3 conv rate=18
        self.block4 = conv_block(in_channels,out_channels,3,1,18,dilation=18)
        #image pooling
        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        upsample_size = (x.shape[-1],x.shape[-2])

        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out5 = self.block5(x)
        out5 = F.interpolate(
            out5,size=upsample_size,mode='bilinear',align_corners=False
        )

        out = torch.cat([out1,out2,out3,out4,out5],dim=1)
        return out
    

class DeepLabV3(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.backbone = ResNet101(in_channels)
        self.aspp = AtrousSpatialPyramidPooling(2048,256)
        self.conv1 = conv_block(256*5,256,1,1,0)
        self.conv2 = nn.Conv2d(256,num_classes,kernel_size=1)
        
        
    def forward(self, x):

        upsample_size = (x.shape[-1],x.shape[-2])

        backbone_out = self.backbone(x)
        aspp_out = self.aspp(backbone_out)
        out = self.conv1(aspp_out)
        out = self.conv2(out)
        out = F.interpolate(
            out,size=upsample_size,mode='bilinear',align_corners=True
        ) 
        return out

from timm.models.layers import trunc_normal_, DropPath
from functools import partial

def conv_block(in_ch,out_ch,k_size,stride,padding,dilation=1,relu=True):
    block = []
    block.append(nn.Conv2d(in_ch,out_ch,k_size,stride,padding,dilation,bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)


# from mmcv_custom import load_checkpoint
# from mmseg.utils import get_root_logger
# from mmseg.models.builder import BACKBONES


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# @BACKBONES.register_module()
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        #1x1 conv
        self.block1 = conv_block(in_channels,out_channels,1,1,0)
        #3x3 conv rate=6
        self.block2 = conv_block(in_channels,out_channels,3,1,6,dilation=6)
        #3x3 conv rate=12
        self.block3 = conv_block(in_channels,out_channels,3,1,12,dilation=12)
        #3x3 conv rate=18
        self.block4 = conv_block(in_channels,out_channels,3,1,18,dilation=18)
        #image pooling
        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,out_channels,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        upsample_size = (x.shape[-1],x.shape[-2])

        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out5 = self.block5(x)
        out5 = F.interpolate(
            out5,size=upsample_size,mode='bilinear',align_corners=False
        )

        out = torch.cat([out1,out2,out3,out4,out5],dim=1)
        return out
    

class DeepLabV3(nn.Module):
    def __init__(self,num_classes=11):
        super().__init__()
        self.backbone = ConvNeXt()
        self.aspp = AtrousSpatialPyramidPooling(2048,256)
        self.conv1 = conv_block(256*5,256,1,1,0)
        self.conv2 = nn.Conv2d(256,num_classes,kernel_size=1)
        
        
    def forward(self, x):

        upsample_size = (x.shape[-1],x.shape[-2])

        backbone_out = self.backbone(x)[3]
        aspp_out = self.aspp(backbone_out)
        out = self.conv1(aspp_out)
        out = self.conv2(out)
        out = F.interpolate(
            out,size=upsample_size,mode='bilinear',align_corners=True
        ) 
        return out

from ocrnet import get_seg_model
from config import config

config_path= '/opt/ml/input/level2-semantic-segmentation-level2-cv-12/code/baseline/seg_hrnetv2_w48.yaml'
cfg = config
cfg.defrost()
cfg.merge_from_file(config_path)
cfg.freeze()

class OCRNet_Hr48(nn.Module):
    def __init__(self,num_classes=11):
        super().__init__()        
        self.backbone = get_seg_model(cfg)
        self.backbone.cls_head = nn.Conv2d(512,num_classes, kernel_size=(1,1),stride=(1,1))
        self.backbone.aux_head[3] = nn.Conv2d(720,num_classes, kernel_size=(1,1), stride=(1,1))
    #@autocast()
    def forward(self, x):
        x = self.backbone(x)
        x = F.interpolate(input=x[0], size=(512, 512), mode='bilinear', align_corners=True)
        return x


class STL(nn.Module):
    def __init__(self):
        super().__init__()
        self.stl = STLNet.STL()

    def forward(self, x):
        x = self.stl(x)
        return x


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