import os
import random
from collections import defaultdict
from enum import Enum
from turtle import width
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

import albumentations as A
from albumentations.pytorch import ToTensorV2




class TestAugmentation: ## testset용
    def __init__(self, resize, **args):
        self.transform = A.Compose([
                            # A.Resize(resize[0], resize[1]),
                            # A.Normalize(),
                            ToTensorV2(),
                            ])

    def __call__(self, image):
        return self.transform(image=image)

class ValAugmentation(TestAugmentation): ## valset용 // TestAugmentation에서 적용한 transform 똑같이 적용됩니다.
    def __init__(self, resize, **args):
        super().__init__(resize, **args)

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)


class BaseAugmentation: 
    def __init__(self, resize, **args):
        self.transform = A.Compose([
                            # A.Resize(resize[0], resize[1]),
                            # A.Normalize(),
                            ToTensorV2(),
                            ])

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)



class Rotate90:
    def __init__(self, resize, **args):
        self.transform = A.Compose([
                            # A.Resize(resize[0], resize[1]),
                            # A.Normalize(),
                            A.RandomRotate90(p=1),
                            ToTensorV2(),
                            ])

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)

class dragon_Augmentation:
    def __init__(self, resize, **args):
        self.transform = A.Compose([
                            # A.Resize(resize[0], resize[1]),
                            A.RandomRotate90(1),
                            A.RandomResizedCrop(height=512,width=512,scale=(0.6,0.9)),
                            
                            A.OneOf([
                                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,val_shift_limit=0.2,p=0.5),
                                A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5)
                            ]),

                            A.OneOf([
                                A.Blur(p=1),
                                A.GaussianBlur(p=1)     
                            ]),
                            A.Normalize(max_pixel_value=1),
                            ToTensorV2(),
                            ])

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)

class BaseDataset(Dataset):
    """COCO format"""
    def __init__(self, data_dir, dataset_path, mode = 'train', transform = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 
                  'Plastic', 'Styrofoam', 'Plastic bag', 'Battery','Clothing']
        self.dataset_path  = dataset_path

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = self.get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())

    def get_classname(self, classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return "None"


class Rch_augmentation:
    def __init__(self, resize, **args):
        self.transform = A.Compose([
                            # A.Resize(resize[0], resize[1]),
                            # A.Normalize(),
                            A.RandomRotate90(p=1),
                            A.RandomBrightnessContrast(p=0.4),
                            A.HueSaturationValue(hue_shift_limit=23, sat_shift_limit=30, val_shift_limit=25, p=0.4),
                            ToTensorV2(),
                            ])

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)