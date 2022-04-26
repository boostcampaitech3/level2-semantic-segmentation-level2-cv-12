import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# from dataset import MaskBaseDataset
from loss import create_criterion

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import label_accuracy_score, add_hist

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# def grid_image(np_images, gts, preds, n=16, shuffle=False):
#     batch_size = np_images.shape[0]
#     assert n <= batch_size

#     choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
#     figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
#     plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
#     n_grid = np.ceil(n ** 0.5)
#     tasks = ["mask", "gender", "age"]
#     for idx, choice in enumerate(choices):
#         gt = gts[choice].item()
#         pred = preds[choice].item()
#         image = np_images[choice]
#         # title = f"gt: {gt}, pred: {pred}"
#         gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
#         pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
#         title = "\n".join([
#             f"{task} - gt: {gt_label}, pred: {pred_label}"
#             for gt_label, pred_label, task
#             in zip(gt_decoded_labels, pred_decoded_labels, tasks)
#         ])

#         plt.subplot(n_grid, n_grid, idx + 1, title=title)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(False)
#         plt.imshow(image, cmap=plt.cm.binary)

#     return figure


def collate_fn(batch):
    return tuple(zip(*batch))

def save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pth'):
    check_point = {'net': model.state_dict()}
    n_try = 0
    while True:
        n_try += 1
        try:
            if n_try == 1:
                os.makedirs(saved_dir, exist_ok=False)
                output_path = os.path.join(saved_dir, file_name)
            else:
                os.makedirs(f'{saved_dir}_{n_try}', exist_ok=False)
                output_path = os.path.join(f'{saved_dir}_{n_try}', file_name)
            break

        except: 
            pass
    print(f"Save model in {output_path}")
    torch.save(model.module.state_dict(), output_path)

def validation(epoch, model, data_loader, criterion, device):
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
        category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 
                  'Plastic', 'Styrofoam', 'Plastic bag', 'Battery','Clothing']
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss, round(acc, 4), round(mIoU, 4)

def train(train_path, val_path, args):
    seed_everything(args.seed)

    # save_dir = increment_path(os.path.join(args.model_dir, args.name))

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
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
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
        weight_decay=1e-6
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    # logger = SummaryWriter(log_dir=save_dir)
    # with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
    #     json.dump(vars(args), f, ensure_ascii=False, indent=4)
    best_loss = 9999999
    best_mIoU = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        hist = np.zeros((11, 11))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (images, masks, _) in pbar:
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
            
            hist = add_hist(hist, masks, outputs, n_class=11)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                pbar.set_description(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
                # print(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], \
                #         Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')

        scheduler.step()

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
                saved_dir = args.saved_dir + '/' + args.model
                save_model(model, saved_dir, f'epoch{epoch+1:04d}.pth')
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    import os
    # load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=21, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='BaseDataset', help='dataset augmentation type (default: BaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[512, 512], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # segmentation
    parser.add_argument('--saved_dir', type=str, default='../saved')
    parser.add_argument('--dataset_path', type=str, default='../../data')
    args = parser.parse_args()
    print(args)

    train_path =  args.dataset_path + '/train.json'
    val_path = args.dataset_path + '/val.json'

    train(train_path, val_path, args)