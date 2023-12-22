# slurm-234211.out : try EPOCH 100, but stopped because of 1 hour limit
# slurm-234354.out : resume from 26, run as 4 hours limit -> epoch 100 완료!

# save epoch,optimizer.state_dict(), model.state_dict(),scheduler.state_dict() in each 5 epoch.
# resume from the last saved epoch by ckpt

# slurm-244236.out

# 1H + 2:52:22.938896  takes for this process

"""# Setting"""
import argparse
import os
import numpy as np
import PIL
from PIL import Image
import sys
import logging
import time
from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import trange
import timm


import medmnist
from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT
from medmnist import Evaluator

from datetime import timedelta
import os
PATH = os.getcwd()

def model_train(model, data_loader, loss_fn, optimizer, device):

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(data_loader)
    for images, labels in progress_bar:
        labels = labels.argmax(dim=1)  # 클래스 인덱스로 변환
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):  # 모델의 출력이 튜플인 경우
            outputs = outputs[0]  # 첫 번째 출력을 사용
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
#        print("Train Check shape ==> ", predicted.shape, labels.shape)
        total += labels.size(0) # BATCH_SIZE
        correct += predicted.eq(labels).sum().item()
#        print(predicted.eq(labels).sum().item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def model_evaluate(model, data_loader, loss_fn, device):

    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader)
        for images, labels in progress_bar:
            labels = labels.argmax(dim=1)  # 클래스 인덱스로 변환
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):  # 모델의 출력이 튜플인 경우
                outputs = outputs[0]  # 첫 번째 출력을 사용
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
#            print("Test Check shape ==> ", predicted.shape, labels.shape)
            total += labels.size(0) # BATCH_SIZE 
            correct += predicted.eq(labels).sum().item()
#            print(predicted.eq(labels).sum().item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main(args):
    start_time = time.time()

    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """# 1.Dataloader"""
    BATCH_SIZE = 128

    """## Med MNIST"""

    import medmnist
    from medmnist import INFO, Evaluator
    import torch.utils.data as data
    import torchvision.transforms as transforms

    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    import medmnist
    from medmnist import INFO, Evaluator
    import torch.utils.data as data
    import torchvision.transforms as transforms

    # Settings
    data_flag = 'chestmnist' # Channel=1 (black/white image)
    # data_flag = 'breastmnist'
    download = True

    gpu_ids = '0, 1'
    resize = True
    as_rgb = True # for 1 channel to 3 channel

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels'] # 1 for chestMNIST
    n_classes = len(info['label']) # 14 for chestMNIST

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    # If we want to resize from 28x28 to 224x224
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    print('==> Preparing data...')
    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

    pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    print(train_dataset)
    print("===================")
    print(test_dataset)

    # montage
    #train_dataset.montage(length=2)

    # Settings
    download = True

    gpu_ids = '0, 1'
    resize = True
    as_rgb = True # for 1 channel to 3 channel

    # preprocessing
    # If we want to resize from 28x28 to 224x224
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    """# 2. Train"""
    """# DeiT Backbone"""

    # Setting
    NUM_EPOCHS = 100
    lr = 0.001
    gamma=0.1 #delaying the learning rate by 0.1
    milestones = [0.5 * NUM_EPOCHS, 0.75 * NUM_EPOCHS] #delaying the learning rate by 0.1 after 50 and 75 epochs.
    BATCH_SIZE = 128
    loss = nn.CrossEntropyLoss()
    model_name = 'DeiT_chest_pretrained'

    model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=False)
    model.head = nn.Linear(192, 14)
    model.head_dist = nn.Linear(192, 14) #14 classes
    model.to(device)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr)#, weight_decay=weight_decay)
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location = device) # when loading checkpoint, set location to device
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        START_EPOCH = ckpt['epoch'] + 1
        print("[INFO] Resume from epoch ", START_EPOCH)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        START_EPOCH = 0
#    model.eval()

    # Multi-GPU
    if (device.type == "cuda") and (torch.cuda.device_count() > 1):
        print("Multi GPU activate")
    else:
        print("Device: ", device)

    print("========= Train Model =========")
    min_loss = np.inf
    total_val_loss, total_val_acc = 0,0
    for epoch in range(START_EPOCH, NUM_EPOCHS, 1): # evaluate every epoch
        train_loss, train_acc = model_train(model, train_loader, loss, optimizer, device)
        val_loss, val_acc = model_evaluate(model, val_loader, loss, device)

        scheduler.step()

        # Avg용 loss, acc += 코드 넣기
        total_val_loss += val_loss
        total_val_acc += val_acc

        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            torch.save(model.state_dict(), f'{PATH}/{model_name}.pth')

        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

        if epoch % 5 == 0:
            torch.save({
                'epoch':epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict':model.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
            }, f'{PATH}/{model_name}.{epoch+1:03d}.pth')
            print(f'[INFO] epoch {epoch+1:02d} model.pth is saved as checkpoint!')
            
    print("-----------------------------")
    total_val_loss = total_val_loss/NUM_EPOCHS
    total_val_acc = total_val_acc/NUM_EPOCHS
    print(f'Avg_val_loss: {total_val_loss:.5f}, Avg_val_acc: {total_val_acc:.5f}')

    test_loss, test_acc = model_evaluate(model, test_loader, loss, device)
    print(f'test_loss: {test_loss:.5f}, test_accuracy: {test_acc:.5f}')

    end_time = time.time()
    torch.save(model.state_dict(), f'{PATH}/ckpt/{model_name}_epoch{epoch+1:02d}.pth')
    print(str(timedelta(seconds=(end_time - start_time))), " takes for this process")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from')

    args = parser.parse_args()
    main(args)