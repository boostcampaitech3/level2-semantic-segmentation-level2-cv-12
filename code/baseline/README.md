# Model

### 참고 링크 (smp)
[segmentation_models.pytorch (github)](https://github.com/qubvel/segmentation_models.pytorch/tree/740dab561ccf54a9ae4bb5bda3b8b18df3790025)

<br>

### 설치
```
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```

<br>

### Architecture
- Unet 
- UnetPlusPlus
- MAnet 
- Linknet 
- FPN 
- PSPNet 
- PAN 
- DeepLabV3 
- DeepLabV3Plus

<br>

### Encoder
- ResNet
- ResNeXt
- ResNeSt
- Res2Ne(X)t
- RegNet(x/y)
- GERNet
- SE-Net
- SK-ResNe(X)t
- DenseNet
- Inception
- EfficientNet
- MobileNet
- DPN
- VGG

<br>

### Example
```
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

model = smp.Unet(                   # choose architecture
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)

# Preparing your data the same way as during weights pre-training may give your better results (higher metric score and faster convergence)
preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
```

<br>