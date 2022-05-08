import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from glob import glob
import albumentations as A

import wandb

category_dict = {0:'Background', 1:'General trash', 2:'Paper', 3:'Paper pack', 4:'Metal', 5:'Glass', 
                  6:'Plastic', 7:'Styrofoam', 8:'Plastic bag', 9:'Battery',10:'Clothing'}

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    # -- wandb setting
    exp_name = args.model + '_' + args.model_path.split('/')[-1].split('.')[0]
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


def load_model(model_path, device):
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
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

@torch.no_grad()
def inference(test_path, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(args.model_path, device).to(device)

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
    csv_path = args.model_path.split('/')
    csv_path[-1] = f'{csv_path[-2]}_{csv_path[-1]}'.replace('.pth', '.csv')
    csv_path = '/'.join(csv_path)
    submission.to_csv(csv_path, index=False)
    print(f'Inference Done!')


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
    parser.add_argument('--model_path', type=str, default='/opt/ml/input/level2-semantic-segmentation-level2-cv-12/saved/drgon_OCRNet_Hr48_fold2/epoch0029_mIoU06535.pth')
    parser.add_argument('--user', type=str)

    args = parser.parse_args()

    test_path = args.dataset_path + '/test.json'

    inference(test_path, args)
