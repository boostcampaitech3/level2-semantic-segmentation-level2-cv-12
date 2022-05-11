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

import ttach as tta


category_dict = {0:'Background', 1:'General trash', 2:'Paper', 3:'Paper pack', 4:'Metal', 5:'Glass', 
                  6:'Plastic', 7:'Styrofoam', 8:'Plastic bag', 9:'Battery',10:'Clothing'}

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    # -- wandb setting
    exp_name = args.model + '_' + args.type + '_tta'
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
                wandb_media = wandb.Image(imgs[4], masks={
                        "predictions" : {
                            "mask_data" : oms[4],
                            "class_labels" : category_dict
                            },
                        })
                mask_list.append(wandb_media) 
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


def model_soup(first, x, device):
    model_cls = getattr(import_module("model"), args.model)
    if args.type == 'mean':
        for j in x:
            for i in first.keys():
                if i.startswith("model.encoder"):
                    first[i] += j[i]

        for i in first.keys():
            if i.startswith("model.encoder"):
                first[i] = first[i] / (len(x)+1)

    elif args.type == 'greedy':
        for j in x:
            for i in first.keys():
                if i.startswith("model.encoder"):
                    first[i] = (first[i] + j[i]) / 2
    
    folder_path = '/opt/ml/input/code/model_soup'
    folder_list = os.listdir(folder_path)
    n = 1
    while True:
        file_name = args.model + '_encoder_' + args.type + '_' + str(n) + '.pth'
        if file_name in folder_list:
            n += 1
        else:
            model_path = folder_path + '/' + file_name
            break    
    print('model_path :', model_path)
    torch.save(first, model_path)

    model = model_cls().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))

@torch.no_grad()
def inference(test_path, first, x, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    model = model_soup(first, x, device).to(device)

    tta_tfms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90([0, 90,]),
            tta.Scale(scales=[1, 2,]),
            # tta.Resize(sizes=[(512, 512), (768, 768), (384, 384)], original_size=(512, 512)),
        ]
    )
    tta_model = tta.SegmentationTTAWrapper(model, tta_tfms, merge_mode="mean")

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

    file_names, preds = test(tta_model, test_loader, device)
    for file_name, string in zip(file_names, preds):
        submission = submission.append({
            "image_id" : file_name, 
            "PredictionString" : ' '.join(str(e) for e in string.tolist())
            }, ignore_index=True)
    folder_path = '/opt/ml/input/code/model_soup'
    folder_list = os.listdir(folder_path)
    n = 1
    while True:
        file_name =  args.model + '_encoder_' + args.type + '_' + str(n) + '.csv'
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
    parser.add_argument('--type', type=str, default='mean')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(512, 512))
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data'))
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/input/data')    
    parser.add_argument('--user', type=str, default='jina')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--dataset', type=str, default='BaseDataset', help='dataset augmentation type (default: BaseDataset)')
    args = parser.parse_args()
    

    ###############################################################
    ###################### pth 파일 불러오기 #######################
    kfold0 = torch.load("/opt/ml/input/code/model_soup/UnetPlusPlus_Efficient5_N_encoder_mean_5.pth")
    kfold1 = torch.load("/opt/ml/input/code/model_soup/UnetPlusPlus_Efficient5_N_encoder_mean_4.pth")
    kfold2 = torch.load("/opt/ml/input/code/model_soup/UnetPlusPlus_Efficient5_N_encoder_mean_3.pth")
    kfold3 = torch.load("/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/UnetPlusPlus_Efficient5_N_jina_aug_13/epoch0079_mIoU06179.pth")
    kfold4 = torch.load("/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/UnetPlusPlus_Efficient5_N_jina_aug_12/epoch0088_mIoU06133.pth")
    ###############################################################
    #### greedy 이용시 first, x 순서대로 mIoU 오름차순 이어야 함 ####
    first = kfold0
    x = [kfold0, kfold1, kfold4, kfold3, kfold2]
    ###############################################################

    test_path = args.dataset_path + '/test.json'
    inference(test_path, first, x, args)