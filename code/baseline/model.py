import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

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