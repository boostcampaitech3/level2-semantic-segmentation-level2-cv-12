import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

def train_transform():
    transform = A.Compose([
                A.Resize(512, 512),
                ToTensorV2(),
                ])
    A.save(transform, '/opt/ml/input/code/tmp/train_transform.json')
    return transform

def val_transform():
    transform = A.Compose([
                ToTensorV2()
                ])
    A.save(transform, '/opt/ml/input/code/tmp/val_transform.json')
    return transform

def test_transform():
    transform = A.Compose([
                           ToTensorV2()
                           ])
    return transform


def make_dict():
    train_trans = '/opt/ml/input/code/tmp/train_transform.json'
    val_trans = '/opt/ml/input/code/tmp/val_transform.json'
    train_dict, val_dict = dict(), dict()
    
    with open(train_trans, 'r') as f:
        train_ = json.load(f)
        for name in train_["transform"]["transforms"]:
            train_dict[name["__class_fullname__"]] = name
    
    with open(val_trans, 'r') as f:
        val_ = json.load(f)
        for name in val_["transform"]["transforms"]:
            val_dict[name["__class_fullname__"]] = name
    return train_dict, val_dict
