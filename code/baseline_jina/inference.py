import os
import random
import torch
from torch import cuda
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
import albumentations as A
from tqdm import tqdm
from importlib import import_module


from dataset import *
from transform import test_transform

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
    parser.add_argument('model_path', type=str, help='train config file path')
    parser.add_argument('--seed', type=int, default=47, help='random seed (default: 47)')
    parser.add_argument('--dataset_path', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data'))
    parser.add_argument('--model_name', type=str, default='FCNResNet50', help='model type (default: FCNResNet50)')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')

    parser.add_argument('--num_workers', type=int, default=1)


    args = parser.parse_args()

    return args

def test(model_path, model_name, dataset_path, batch_size, seed, num_workers, device):
    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # seed
    seed_everything(seed)
    
    # dataset
    test_path = dataset_path + '/test.json'
    test_trans = test_transform()
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_trans)
    
    # dataloader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          collate_fn=collate_fn)

    # model
    model_module = getattr(import_module("model"), model_name)  # default: FCNResnet50
    model = model_module(num_classes=11)
    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
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
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/input/code/submission/sample_submission.csv', index_col=None)
    
    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)
    
    # submission.csv로 저장
    submission.to_csv("/opt/ml/input/code/submission/{}_best_model.csv".format(model_name), index=False)    

    return file_names, preds_array


def main(args):
    test(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)