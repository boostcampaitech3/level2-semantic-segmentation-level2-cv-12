import argparse
import glob,time
import json
import multiprocessing
import os
import random,math
import re
from importlib import import_module




import numpy as np
import torch

from torch.utils.data import DataLoader


from dataset import CustomDataLoader,AlbumAugmentation
from loss import create_criterion


from utils import *

saved_dir = './saved'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)


def validation(epoch, model, data_loader, criterion, device,sorted_df):
    print(f'Start validation #{epoch}')
    model.eval()

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
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , sorted_df['Categories'])]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss

def train(args):
    set_seed(args.seed)
    
    dataset_path  = '/opt/ml/input/data'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    train_dataset = CustomDataLoader(data_dir=train_path, mode='train')
    train_dataset.set_transform(AlbumAugmentation)
    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path, mode='val')
    val_dataset.set_transform(AlbumAugmentation)

    check_df = train_dataset.re_df()
    
    train_loader = DataLoader(dataset=train_dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=collate_fn)

    val_loader = DataLoader(dataset=val_dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=collate_fn)

    model = getattr(import_module("model"),args.model)()()

    criterion = create_criterion(args.criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-6
    )

    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    # device 할당
    model = model.to(device)
    for epoch in range(args.epochs):
        model.train()

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            
            
            
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
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % 1 == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device,check_df)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, saved_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 1)')
    
    
    
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
    
    parser.add_argument('--model',type= str,default='Fcn_Resnet50',help = "select model")
    
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 1e-3)')
    
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    

    args = parser.parse_args()
    print(args)

    
    train(args)