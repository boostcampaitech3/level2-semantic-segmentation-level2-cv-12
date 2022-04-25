import os
import os.path as osp
import random
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from importlib import import_module

from dataset import *
from utils import *
from transform import *
from loss import *

import wandb

def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset_path', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data'))
    parser.add_argument('--model_name', type=str, default='FCNResNet50', help='model type (default: FCNResNet50)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--optim', type=str, default='AdamW', help='optimizer type (default: AdamW)')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--saved_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/input/code/saved'))

    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()

    return args

category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack',
 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def train(dataset_path, model_name, criterion, optim, saved_dir, learning_rate, val_every, device, seed, num_epochs, batch_size, num_workers):
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # seed
    seed_everything(seed)
    
    # dataset
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    train_trans = train_transform()
    val_trans = val_transform()
    # transform config
    train_dict, val_dict = make_dict()

    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_trans)
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_trans)
    
    # dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn)

    # model
    model_module = getattr(import_module("model"), model_name)  # default: FCNResnet50
    model = model_module(num_classes=11)
    
    # Loss function 정의
    criterion = create_criterion(criterion)  # default: cross_entropy

    # optimizer
    opt_module = getattr(import_module("torch.optim"), optim) 
    optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=1e-6)    # default: AdamW
    
    # scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)

    # wandb setting
    config = {'model':model_name, 'batch_size':batch_size, 'optimizer': optim, 'learning_rate':learning_rate, 
    'epochs':num_epochs, 'seed':seed, 'train_transform':train_dict, 'val_transform':val_dict}
    wandb_name = 'jina_' + f'{model_name}' + '_' + f'{optim}' + '_lr_' + f'{learning_rate}'
    wandb.init(entity = 'yolo12', project = 'segmentation', name = wandb_name, config = config)
    wandb.watch(model, log=all)
    
    # 모델 저장 함수 정의
    if not os.path.isdir(saved_dir):                                                           
        os.mkdir(saved_dir)

    def save_model(model, saved_dir):
        file_name=f'{model_name}_best_model.pt'
        check_point = {'net': model.state_dict()}
        output_path = os.path.join(saved_dir, file_name)
        torch.save(model, output_path)
    
    # train
    print(f'Start training..')
    
    n_class = 11
    best_mIoU = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            # device 할당
            model = model.to(device)
            
            # inference
            outputs = model(images)['out']
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                wandb.log({ "train/loss": loss.item(), 
                            "train/mIoU": mIoU,
                            })

        print('Elapsed time: {}'.format(timedelta(seconds=time.time() - epoch_start)))
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            val_mIoU = validation(epoch + 1, model, val_loader, criterion, device)
            if val_mIoU > best_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_mIoU = val_mIoU
                save_model(model, saved_dir)


def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(tqdm(data_loader)):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}')
        print('IoU by class :')
        for class_iou in IoU_by_class:
            print(class_iou)
        wandb.log({ "val/loss": avrg_loss.item(), 
                    "val/accuracy": acc,
                    "val/mIoU": mIoU,
                    })
        
    return mIoU

def main(args):
    train(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)