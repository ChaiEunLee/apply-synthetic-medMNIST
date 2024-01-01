# Created by Park Sehyun (reference from Lee Chai Eun's code)


"""# Setting"""
import argparse
import numpy as np
import time
import pprint
import datetime

import PIL
from PIL import Image

import sys
import logging

from collections import OrderedDict
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms, datasets
from tensorboardX import SummaryWriter
from tqdm import trange
import timm


def create_log_func(path):
    f = open(path, 'a')
    counter = [0]
    def log(txt, color=None):
        f.write(txt + '\n')
        if color == 'red': txt = "\033[91m {}\033[00m" .format(txt)
        elif color == 'green': txt = "\033[92m {}\033[00m" .format(txt)
        elif color == 'violet': txt = "\033[95m {}\033[00m" .format(txt)
        print(txt)
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return log, f.close

#Time 
now = datetime.datetime.utcnow()
time_gap = datetime.timedelta(hours=9)
now += time_gap
now = now.strftime("%Y%m%d_%H%M%S") 

# Set ratio
RATIO = 0
model_name = 'ResNet_pneu_synthetic_0_chestMNIST'
project_name = 'transfer_learning'

import os
PATH = os.getcwd()
os.chdir('..')
PREPATH = os.getcwd() # for dataset and pretrained model
HOME = '/home/sehyunpark/project_MedTransfer/'

if not os.path.exists(f"./{project_name}/report/transfer_{model_name}/"):
    os.makedirs(f"./report/{project_name}/transfer_{model_name}/")
if not os.path.exists(f"./{project_name}/checkpoint/transfer_{model_name}/"):
    os.makedirs(f"./{project_name}/checkpoint/transfer_{model_name}/")
    
log_path = f"./{project_name}/checkpoint/transfer_{model_name}/{now}.log"
log, log_close = create_log_func(log_path)          # Abbreviate log_func to log
with open(log_path, 'w+') as f:
    pprint.pprint(f)

log('******************************          Arguments          ******************************')
log(f'Project name: {project_name}, at time {now}')

log(f'Model: {model_name}')

import random
random.seed(2023)
torch.manual_seed(2023)

from datetime import timedelta
start_time = time.time()
# Check cuda
#torch.cuda.is_available()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
log(f'Device: {device}')

"""# 1.Dataloader"""
BATCH_SIZE = 128

"""## Med MNIST"""
import medmnist
from medmnist import INFO, Evaluator
from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT
from medmnist import Evaluator
print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# Settings
data_flag = 'pneumoniamnist' # Channel=1 (black/white image)
download = True

resize = True
as_rgb = True # for 1 channel to 3 channel

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels'] # 1 for pneumoniaMNIST, 14 for chestMNIST
n_classes = len(info['label']) 

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
#pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

print("=====> Med MNIST data... DONE")
print(train_dataset)
print("===================")
print(test_dataset)

"""## Synthetic data"""
download = True

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

#!unzip -qq f"{PATH}/synthetic.zip"
#import zipfile
#with zipfile.ZipFile(f"{PATH}/synthetic_shared.zip", 'r') as zip_ref:
#    zip_ref.extractall(f"{PATH}")


# load dataset
class SyntheticDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.label_file = label_file
        self.transform = transform

        self.data = []
        with open(self.data_dir+label_file, 'r') as file:
            for line in file:
                filename, label = line.strip().split()
                self.data.append((filename, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename, label = self.data[index]
        img_path = os.path.join(self.data_dir, filename)
        img_path = img_path.replace('..', os.getcwd())

        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, label

# load the data
data_dir = f'{HOME}/synthetic/'
label_file = 'label.txt'
dataset = SyntheticDataset(data_dir, label_file, transform = data_transform)
dataset_size = len(dataset)
train_size = int(dataset_size*0.8)
val_size = dataset_size - train_size
train_dataset_synthetic, val_dataset_synthetic = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(2023))

# Check number of each dataset size
print("=====> Synthetic data... DONE")
print(f"Synthetic training dataset size : {len(train_dataset_synthetic)}")
print(f"Synthetic validation dataset size : {len(val_dataset_synthetic)}")

# Dataloaders
train_loader_synthetic = DataLoader(train_dataset_synthetic, batch_size=BATCH_SIZE, shuffle=True)
val_loader_synthetic = DataLoader(val_dataset_synthetic, batch_size=BATCH_SIZE, shuffle=False)


""" Concat dataset """
# Choose the ratio - 100%
print("=====> Ratio: ", RATIO)
log("=====> Ratio: ", RATIO)
if RATIO == 100:
    # Train
    train_size = len(train_dataset)
    else_size = len(train_dataset_synthetic) - train_size
    train_dataset_synthetic, _ = random_split(train_dataset_synthetic, [train_size, else_size], generator=torch.Generator().manual_seed(2023))

    # Validation
    val_size = len(val_dataset)
    else_size = len(val_dataset_synthetic) - val_size
    val_dataset_synthetic, _ = random_split(val_dataset_synthetic, [val_size, else_size], generator=torch.Generator().manual_seed(2023))

elif RATIO == 200:
    # Train
    train_size = len(train_dataset)*2
    else_size = len(train_dataset_synthetic) - train_size
    train_dataset_synthetic, _ = random_split(train_dataset_synthetic, [train_size, else_size], generator=torch.Generator().manual_seed(2023))

    # Validation
    val_size = len(val_dataset)*2
    else_size = len(val_dataset_synthetic) - val_size
    val_dataset_synthetic, _ = random_split(val_dataset_synthetic, [val_size, else_size], generator=torch.Generator().manual_seed(2023))

elif RATIO == 300:
    # Train
    train_size = len(train_dataset)*3
    else_size = len(train_dataset_synthetic) - train_size
    train_dataset_synthetic, _ = random_split(train_dataset_synthetic, [train_size, else_size], generator=torch.Generator().manual_seed(2023))

    # Validation
    val_size = len(val_dataset)*3
    else_size = len(val_dataset_synthetic) - val_size
    val_dataset_synthetic, _ = random_split(val_dataset_synthetic, [val_size, else_size], generator=torch.Generator().manual_seed(2023))


print("===> ... Concat Dataset...")
train_dev_sets = ConcatDataset([train_dataset, train_dataset_synthetic])
val_dev_sets = ConcatDataset([val_dataset, val_dataset_synthetic])

train_dev_loader = DataLoader(dataset=train_dev_sets, batch_size=BATCH_SIZE, shuffle=True)
val_dev_loader = DataLoader(dataset=val_dev_sets, batch_size=BATCH_SIZE, shuffle=True)

print("=====> ... Prepare Dataset DONE ...")
print(f"Training dataset size : {len(train_dev_sets)}")
print(f"Validation dataset size : {len(val_dev_sets)}")
print(f"Test dataset size : {test_dataset.info['n_samples']['test']}") # only pneumonia MNIST test is test data

"""# 2. Train"""

"""# 3) Synthetic fine-tune model """
# check time
model_start_time = time.time()

log(f'==> Train {model_name}...')

# Load backbone model
# model = models.resnet50(weights="IMAGENET1K_V2")
model = models.resnet50()

# Fine-tuning
last_layer = model.fc
in_features = last_layer.in_features

# for binary classification
model.fc= nn.Linear(in_features, 2)
# pth 불러오기
ckpt = torch.load(f'{PATH}/resnet50_224_1.pth')
new_state_dict = {}
for key, value in ckpt['net'].items():
    # new_state_dict[new_key] = value
    if 'fc' in key:
        continue
    else:
        new_state_dict[key] = value  
model.load_state_dict(new_state_dict, strict=False)  
model.to(device)

# Setting
NUM_EPOCHS = 50
lr = 0.0005 * BATCH_SIZE/512
weight_decay = 0.05
warmup_steps = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps) # learning rate scheduler
loss = nn.CrossEntropyLoss()

# Multi-GPU
if (device.type == "cuda") and (torch.cuda.device_count() > 1):
    print("Multi GPU activate")
else:
    print("Device: ", device)


def model_train(model, data_loader, loss_fn, optimizer, device):

    model.train()
    running_loss = 0
    corr = 0

    prograss_bar = tqdm(data_loader)
    for img, lbl in prograss_bar:
        lbl = lbl.type(torch.LongTensor)
#        lbl = lbl.type(torch.FloatTensor)
        img, lbl = img.to(device), lbl.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, lbl.squeeze(dim=-1)) # input:output[0], target: lbl
        #print("Check shape of loss ==>", output[0].shape, lbl.squeeze(dim=-1).shape)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        _, pred = output.max(dim=1)
        corr += pred.eq(lbl.squeeze(dim=-1)).sum().item()
        #print("Check shape ==> ", pred.shape, lbl.squeeze(dim=-1).shape)
        #print(corr)
        running_loss += loss.item() * img.size(0)

    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn, device):

    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        prograss_bar = tqdm(data_loader)

        for img, lbl in prograss_bar:
          lbl = lbl.type(torch.LongTensor)
          img, lbl = img.to(device), lbl.to(device)

          output = model(img) #(batch, num_classes)
          _, pred = output.max(dim=1)

          corr += torch.sum(pred.eq(lbl.squeeze(dim=-1))).item()
          #print("Check shape ==> ", pred.shape, lbl.squeeze(dim=-1).shape)
          #print(corr)
          running_loss += loss_fn(output,lbl.squeeze(dim=-1)).item() * img.size(0)

        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc

log("========= Train Model =========")
min_loss = np.inf
total_val_loss, total_val_acc = 0,0
for epoch in range(0, NUM_EPOCHS, 1): # evaluate every epoch
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

    # print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
    log(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
print("-----------------------------")
log("-----------------------------")
total_val_loss = total_val_loss/NUM_EPOCHS
total_val_acc = total_val_acc/NUM_EPOCHS
# print(f'Avg_val_loss: {total_val_loss:.5f}, Avg_val_acc: {total_val_acc:.5f}')
log(f'Avg_val_loss: {total_val_loss:.5f}, Avg_val_acc: {total_val_acc:.5f}')

model_end_time = time.time()
print(str(timedelta(seconds=(model_end_time - model_start_time))), " takes for training ", model_name, "!")
time_duration = str(timedelta(seconds=(model_end_time - model_start_time)))
log(f"{time_duration} takes for training {model_name} !")


"""# 3. Evaluation """
log('====== Evaluation by checkpoint======')
""" # 1) Deit Base """
# If stopped, bring it from checkpoint
# model 불러오기
# model = models.resnet50(weights="IMAGENET1K_V2")
model = models.resnet50()

# Fine-tuning
last_layer = model.fc
in_features = last_layer.in_features

# for binary classification
model.fc= nn.Linear(in_features, 2)
# pth 불러오기
model.load_state_dict(torch.load(f'{PATH}/{model_name}.pth')) 
# model.load_state_dict(torch.load(f'{PATH}/ResNet_pneu_synthetic_361.pth')) 
log(f'Chekpoint: {model_name}.pth')
model.to(device)
#model.eval()

def model_evaluate(model, data_loader, loss_fn, device):

    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        prograss_bar = tqdm(data_loader)

        for img, lbl in prograss_bar:
          lbl = lbl.type(torch.LongTensor)
          img, lbl = img.to(device), lbl.to(device)

          output = model(img) #(batch, num_classes)
          _, pred = output.max(dim=1)
          #print("Check shape ==> ", pred.shape,  lbl.squeeze(dim=-1).shape)

          corr += torch.sum(pred.eq(lbl.squeeze(dim=-1))).item()
          running_loss += loss_fn(output,lbl.squeeze(dim=-1)).item() * img.size(0)

        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc


# model
# test_loader
loss = nn.CrossEntropyLoss()
val_loss, val_acc = model_evaluate(model, test_loader, loss, device)
log(f'test_loss: {val_loss:.5f}, test_accuracy: {val_acc:.5f}')
