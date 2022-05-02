import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from tkinter import E

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from torch.utils.data import DataLoader

from loss import create_criterion

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import label_accuracy_score, add_hist

from loss import *
from kornia.losses import focal_loss

import wandb

category_dict = {0:'Background', 1:'General trash', 2:'Paper', 3:'Paper pack', 4:'Metal', 5:'Glass', 
                  6:'Plastic', 7:'Styrofoam', 8:'Plastic bag', 9:'Battery',10:'Clothing'}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    return tuple(zip(*batch))

def make_dir(saved_dir):
    n_try = 0
    while True:
        n_try += 1
        try:
            if n_try == 1:
                os.makedirs(saved_dir, exist_ok=False)
                return saved_dir
            else:
                os.makedirs(f'{saved_dir}_{n_try}', exist_ok=False)
                return f'{saved_dir}_{n_try}'

        except: 
            pass
        
def save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pth'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    print(f"Save model in {output_path}")
    torch.save(model.module.state_dict(), output_path)

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
            
            # wandb image
            if step % 5 == 0:
                wandb_media = wandb.Image(images[0], masks={
                        "predictions" : {
                            "mask_data" : outputs[0],
                            "class_labels" : category_dict
                            },
                        "ground_truth" : {
                            "mask_data" : masks[0],
                            "class_labels" : category_dict}
                        })
                mask_list.append(wandb_media)
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 
                  'Plastic', 'Styrofoam', 'Plastic bag', 'Battery','Clothing']
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        for _dict in IoU_by_class:
            wandb.log({f'class/{k}_mIoU' : v for k, v in _dict.items()})
        wandb.log({ "val/loss": avrg_loss.item(), 
                    "val/accuracy": acc,
                    "val/mIoU": mIoU,
                    })

    wandb.log({"val" : mask_list})

    return avrg_loss, round(acc, 4), round(mIoU, 4)

def train(train_path, val_path, args):
    seed_everything(args.seed)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    train_transform = transform_module(
        resize=args.resize,
        )

    val_transform_module = getattr(import_module("dataset"), 'ValAugmentation')
    val_transform = val_transform_module(
        resize=args.resize,
        )

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseDataset
    train_dataset = dataset_module(
        data_dir=train_path,
        dataset_path = args.dataset_path,
        mode='train',
        transform=train_transform
    )

    val_dataset = dataset_module(
        data_dir=val_path,
        dataset_path = args.dataset_path,
        mode='val',
        transform=val_transform
    )

    # -- data_loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            drop_last=True,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=4,
                                            drop_last=True,
                                            collate_fn=collate_fn)

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module().to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
   
    criterion = create_criterion(args.criterion)  # default: cross_entropy

    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    opt_scheduler = getattr(import_module("torch.optim.lr_scheduler"),args.scheduler)
    if args.scheduler == 'StepLR':
        scheduler = opt_scheduler(optimizer,args.lr_decay_step,gamma=0.5)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = opt_scheduler(optimizer, 'min',verbose=False)
    elif args.scheduler == 'CosineAnnealingLR':
        #인자값들은 임시로 변경해서 사용하시면 됩니다.
        scheduler = opt_scheduler(optimizer, T_max=100, eta_min=0 ,last_epoch=-1, verbose=False )

    # -- config & directory settings
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f'{args.model}_{args.augmentation}'
    saved_dir = f'{args.saved_dir}/{exp_name}'
        
    saved_dir = make_dir(saved_dir)
    config = args.__dict__.copy()
    config['augmentation'] = {args.augmentation : train_transform.transform._to_dict()['transforms']}
    config['model'] = {args.model : model.__str__().split('\n')}
    with open(f'{saved_dir}/{exp_name}_config.txt', mode='w') as f:
        json.dump(config, f, indent=2)

    
    # -- wandb setting
    wandb_name = f'{args.user}_{exp_name}'
    wandb.init(entity = 'yolo12', project = 'segmentation', name = wandb_name, config = config)
    wandb.watch(model, log=all)

    best_loss = 9999999
    best_mIoU = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        hist = np.zeros((11, 11))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        mask_list = []
        for step, (images, masks, _) in pbar:
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            if args.model in ['BaseModel', 'BaseModel2', 'FCNResnet101', 'Deeplabv3_Resnet50', 'Deeplabv3_Resnet101']:
                outputs = model(images)['out']
            else:
                outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            # wandb image
            if step % 5 == 0:
                wandb_media = wandb.Image(images[0], masks={
                        "predictions" : {
                            "mask_data" : outputs[0],
                            "class_labels" : category_dict
                            },
                        "ground_truth" : {
                            "mask_data" : masks[0],
                            "class_labels" : category_dict}
                        })
                mask_list.append(wandb_media)
            if (step + 1) % args.log_interval == 0:
                hist = add_hist(hist, masks, outputs, n_class=11)
                acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
                lr_rate = optimizer.param_groups[0]['lr']
                # step 주기에 따른 loss 출력
                # pbar.set_description(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], lr: {scheduler.get_last_lr()[0]}, Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                pbar.set_description(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], lr: {lr_rate}, Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                wandb.log({ "train/loss": loss.item(),
                            "train/mIoU": mIoU,
                            })        
        wandb.log({"train" : mask_list})

        # validation 주기에 따른 loss 출력 및 best model 저장
        val_every = 1
        if (epoch + 1) % val_every == 0:
            avrg_loss, acc, mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            # if avrg_loss < best_loss:
            #     print(f"Best performance at epoch: {epoch + 1}")
            #     best_loss = avrg_loss
            #     saved_dir = args.saved_dir + '/' + args.model
            #     print(f"Save model in {saved_dir}")
            #     save_model(model, saved_dir, f'epoch{epoch:04d}.pth')
            if mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                best_mIoU = mIoU
                save_model(model, saved_dir, f'epoch{epoch+1:04d}_mIoU{str(best_mIoU).replace(".","")}.pth')
            print()
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(avrg_loss)
        else:
            scheduler.step()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=21, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='BaseDataset', help='dataset augmentation type (default: BaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", type=eval, default='[512, 512]', help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--exp_name', type=str, default='')

    # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # segmentation
    parser.add_argument('--saved_dir', type=str, default='../saved')
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/input/data')
    
    # wandb
    parser.add_argument('--user', type=str)    

    # fold
    parser.add_argument('--fold', type=int, default=0)
    #schedeuler
    parser.add_argument('--scheduler',type=str,default='StepLR')
    parser.add_argument('--accumulate_mode',type=bool,default=False)
    args = parser.parse_args()
    print(args)

    train_path =  args.dataset_path + f'/train_fold{args.fold}.json'
    val_path = args.dataset_path + f'/val_fold{args.fold}.json'

    train(train_path, val_path, args)
