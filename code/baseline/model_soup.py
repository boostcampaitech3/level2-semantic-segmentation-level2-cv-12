from importlib import import_module
import argparse
from copy import deepcopy

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from glob import glob
import albumentations as A

from loss import *
from utils import label_accuracy_score, add_hist

import wandb

category_dict = {0:'Background', 1:'General trash', 2:'Paper', 3:'Paper pack', 4:'Metal', 5:'Glass', 
                  6:'Plastic', 7:'Styrofoam', 8:'Plastic bag', 9:'Battery',10:'Clothing'}

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    # -- wandb setting
    exp_name = args.model + '_' + args.type
    wandb_name = f'{args.user}_{exp_name}'
    wandb.init(entity = 'yolo12', project = 'seg_inference', name = wandb_name)
    wandb.watch(model, log=None)
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    mask_list = []
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):
            
            # inference (512 x 512)
            if args.model in ['BaseModel', 'BaseModel2', 'FCNResnet101', 'Deeplabv3_Resnet50', 'Deeplabv3_Resnet101']:
                outs = model(torch.stack(imgs).to(device))['out']
            else:
                outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # wandb image
            if step % 5 == 0:
                wandb_media = wandb.Image(imgs[5], masks={
                        "predictions" : {
                            "mask_data" : oms[5],
                            "class_labels" : category_dict
                            },
                        })
                mask_list.append(wandb_media)            
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    
        wandb.log({'inference' : mask_list}) 
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()
    mask_list = []
    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            if args.model in ['BaseModel', 'BaseModel2', 'FCNResnet101', 'Deeplabv3_Resnet50', 'Deeplabv3_Resnet101']:
                outputs = model(images)['out']
            else:
                outputs = model(images)
            
             # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 
                  'Plastic', 'Styrofoam', 'Plastic bag', 'Battery','Clothing']
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')

    return avrg_loss, round(acc, 4), round(mIoU, 4)


def add_ingredient(soup, i, ckp_i, device):
    ingredient = getattr(import_module("model"), args.model)
    ingredient = ingredient().to(device)
    checkpoint = torch.load(ckp_i, map_location=device)
    ingredient.load_state_dict(checkpoint)
    for param1, param2 in zip(soup.parameters(), ingredient.parameters()):
        param1.data = (param1.data + param2.data) / 2
        
    return soup

def greedy_soup(ckp_list, device):
    criterion = create_criterion(args.criterion)  # default: cross_entropy

    val_path = '/opt/ml/input/data/val_fold4.json'
    # -- augmentation
    val_transform_module = getattr(import_module("dataset"), 'ValAugmentation')
    val_transform = val_transform_module(
        resize=args.resize,
        )

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseDataset
    val_dataset = dataset_module(
        data_dir=val_path,
        dataset_path = args.dataset_path,
        mode='val',
        transform=val_transform
    )

    # -- data_loader
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=4,
                                            drop_last=True,
                                            collate_fn=collate_fn)

    soup_ = getattr(import_module("model"), args.model)
    soup = soup_().to(device)
    checkpoint = torch.load(ckp_list[0], map_location=device)
    soup.load_state_dict(checkpoint)
    
    avrg_loss, acc, best_mIoU = validation(1, soup, val_loader, criterion, device)
    # best_acc, loss = test(soup, testloader)
    print(f'first acc: {best_mIoU:.2f}%')
    
    # cook
    for i, ckp_i in enumerate(ckp_list[1:]):
        soup_next = deepcopy(soup)
        soup_next = add_ingredient(soup_next, i, ckp_i, device)
        
        avrg_loss, acc, mIoU = validation(i+2, soup_next, val_loader, criterion, device)
        if mIoU > best_mIoU:
            print(f"Best performance at epoch: {i + 2}")
            best_mIoU = mIoU
            soup = soup_next
    return soup


def add_uniform(soup, i, ckp_i, device):
    ingredient = getattr(import_module("model"), args.model)
    ingredient = ingredient().to(device)
    checkpoint = torch.load(ckp_i, map_location=device)
    ingredient.load_state_dict(checkpoint)
    for param1, param2 in zip(soup.parameters(), ingredient.parameters()):
        param1.data = (param1.data * (i+1) + param2.data) / (i+2)
    return soup

def uniform_soup(ckp_list, device):
    soup_ = getattr(import_module("model"), args.model)
    soup = soup_().to(device)
    checkpoint = torch.load(ckp_list[0], map_location=device)
    soup.load_state_dict(checkpoint)

    for i, ckp_i in enumerate(ckp_list[1:]):
        soup_next = deepcopy(soup)
        soup_next = add_uniform(soup_next, i, ckp_i, device)
        soup = soup_next

    return soup


def collate_fn(batch):
    return tuple(zip(*batch))

@torch.no_grad()
def inference(test_path, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.type == 'uniform':
        model = uniform_soup(ckp_list, device).to(device)
    else:
        model = greedy_soup(ckp_list, device).to(device)
    
    # transform
    transform_module = getattr(import_module("dataset"), 'TestAugmentation') 
    test_transform = transform_module(
        resize=args.resize,
        )

    # dataset
    dataset_module = getattr(import_module("dataset"), 'BaseDataset') 
    test_dataset = dataset_module(
        data_dir=test_path,
        dataset_path = args.dataset_path,
        mode='test',
        transform=test_transform
    )

    # dataloader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    print("Calculating inference results..")

    submission = pd.read_csv('/opt/ml/input/level2-semantic-segmentation-level2-cv-12/code/baseline/sample_submission.csv', index_col=None)

    file_names, preds = test(model, test_loader, device)
    for file_name, string in zip(file_names, preds):
        submission = submission.append({
            "image_id" : file_name, 
            "PredictionString" : ' '.join(str(e) for e in string.tolist())
            }, ignore_index=True)
    folder_path = '/opt/ml/input/code/model_soup'
    folder_list = os.listdir(folder_path)
    n = 1
    while True:
        file_name = args.model + '_' + args.type + str(n) + '.csv'
        if file_name in folder_list:
            n += 1
        else:
            csv_path = folder_path + '/' + file_name
            break
    submission.to_csv(csv_path, index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='UnetPlusPlus_Efficient5_N', help='model type (default: BaseModel)')
    parser.add_argument('--type', type=str, default='greedy')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(512, 512))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data'))
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/input/data')    
    parser.add_argument('--user', type=str, default='jina')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--dataset', type=str, default='BaseDataset', help='dataset augmentation type (default: BaseDataset)')
    args = parser.parse_args()
    
    ckp_list = ['/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/Deeplabv3Plus_SEResnet152_jina_aug_6/epoch0029_mIoU06155.pth',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/Deeplabv3Plus_SEResnet152_jina_aug_6/epoch0030_mIoU06233.pth',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/Deeplabv3Plus_SEResnet152_jina_aug_6/epoch0035_mIoU06262.pth',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/Deeplabv3Plus_SEResnet152_jina_aug_6/epoch0051_mIoU06302.pth',
    '/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/Deeplabv3Plus_SEResnet152_jina_aug_6/epoch0053_mIoU0636.pth',
    ]

    test_path = args.dataset_path + '/test.json'
    inference(test_path, args)
