import argparse
import os
from importlib import import_module
from pickletools import optimize

import pandas as pd
import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from glob import glob
import albumentations as A
import torch.nn as nn
import torch.nn.functional as F
import wandb

from tqdm import tqdm
from utils import label_accuracy_score, add_hist

category_dict = {0:'Background', 1:'General trash', 2:'Paper', 3:'Paper pack', 4:'Metal', 5:'Glass', 
                  6:'Plastic', 7:'Styrofoam', 8:'Plastic bag', 9:'Battery',10:'Clothing'}

def load_model(model_path, device,self_train=True):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls().to(device)
    
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)
    print('====================================================================================================================')
    print('====================================================================================================================')
    print(f'load checkpoint from {model_path}')
    print('====================================================================================================================')
    print('====================================================================================================================')
    if self_train:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    else:
        model.train()
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pth'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    print(f"Save model in {output_path}")
    torch.save(model.module.state_dict(), output_path)
    
def knowlege_distillation_loss(logits,teacher_logits):
	alpha = 0.1
	T = 10
	
	student_loss = F.cross_entropy(input=logits,target=teacher_logits)
	distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T,dim=1),F.softmax(teacher_logits/T,dim=1)) *(T*T)
	total_loss = alpha*student_loss + (1-alpha)*distillation_loss
	
	return total_loss

def inference(test_path, args):
    saved_dir = '/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    teacher = load_model(args.model_path, device).to(device)
    student = load_model(args.model_path,device,self_train=False)
    
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student.parameters(),lr=6e-05)


    n_class = 11
    total_loss = 0
    cnt = 0
    
    hist = np.zeros((n_class, n_class))
    for epoch in range(200):
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):  
            # inference (512 x 512)
            if args.model in ['BaseModel', 'BaseModel2', 'FCNResnet101', 'Deeplabv3_Resnet50', 'Deeplabv3_Resnet101']:
                teacher_outs = teacher(torch.stack(imgs).to(device))['out']
                teacher_outs
            else:
                teacher_outs = teacher(torch.stack(imgs).to(device))
                teacher_outs
            
            # inference (512 x 512)
            if args.model in ['BaseModel', 'BaseModel2', 'FCNResnet101', 'Deeplabv3_Resnet50', 'Deeplabv3_Resnet101']:
                student_outs = student(torch.stack(imgs).to(device))['out']
            else:
                student_outs = student(torch.stack(imgs).to(device))
        
            # loss 계산 (cross entropy loss)
            loss = knowlege_distillation_loss(student_outs, teacher_outs)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            student_outs = torch.argmax(student_outs, dim=1).detach().cpu().numpy()
            teacher_outs = torch.argmax(teacher_outs, dim=1).detach().cpu().numpy()
        
            best_mIoU = 0
            hist = add_hist(hist, teacher_outs, student_outs, n_class=11)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            avrg_loss = total_loss / cnt
        print(f'Epoch [{epoch+1}/{200}], Step [{epoch+1}/{len(test_loader)}], Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
        if mIoU > best_mIoU:
            print(f"Best performance at epoch: {epoch + 1}")
            best_mIoU = mIoU
            save_model(student, saved_dir, f'epoch{epoch+1:04d}_mIoU{str(best_mIoU).replace(".","")}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(512, 512))
    parser.add_argument('--model', type=str, default='OCRNet_Hr48', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data'))

    # segmentation
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/input/data')
    parser.add_argument('--model_path', type=str, default='/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/drgon_OCRNet_Hr48_fold2_aug/epoch0074_mIoU06578.pth')

    parser.add_argument('--user', type=str)

    args = parser.parse_args()

    test_path = args.dataset_path + '/test.json'

    inference(test_path, args)