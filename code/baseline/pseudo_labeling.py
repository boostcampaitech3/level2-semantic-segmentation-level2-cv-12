import warnings
warnings.filterwarnings('ignore')

import torch
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import *
from dataset import *

import argparse


def collate_fn(batch):
    return tuple(zip(*batch))


def test(model, data_loader, device):
    size = 512
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    
        # wandb.log({'inference' : mask_list}) 
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

def pseudo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_path = "/opt/ml/input/data/test.json"
    test_transform = A.Compose([
                            A.Resize(512, 512),
                            ToTensorV2()
                            ])
    dataset_path = '/opt/ml/input/data'

    test_dataset = BaseDataset(data_dir=test_path, dataset_path = dataset_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=8,
                                                shuffle=False,
                                                num_workers=4,
                                                collate_fn=collate_fn)

    ##### best mIoU pth #####
    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location=device)
    model = OCRNet_Hr48().to(device)
    model.load_state_dict(checkpoint)
    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/input/level2-semantic-segmentation-level2-cv-12/code/baseline/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    ##### image name을 npy로 저장
    name_list = []
    for i in file_names:
        name_list.append(i)
    np.save('/opt/ml/input/data/img_name', name_list)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)
        
        

    # submission.csv로 저장
    submission.to_csv("/opt/ml/input/data/submission_test.csv", index=False)

    df = pd.read_csv('/opt/ml/input/data/submission_test.csv')

    mask_path = '/opt/ml/input/data/mask'
    if os.path.isdir('/opt/ml/input/data/mask'):
        pass
    else:
        os.mkdir(mask_path)

    for i in range(len(df)):
        cur = df.iloc[i, :]
        arr = np.array(cur['PredictionString'].split()).reshape(512, 512)
        np.save(mask_path + '/' + format(i, '03') , arr)

    print(len(arr[0]),len(arr[220]),len(arr[511]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/drgon_OCRNet_Hr48_fold2_aug/epoch0074_mIoU06578.pth')
    args = parser.parse_args()

    pseudo()