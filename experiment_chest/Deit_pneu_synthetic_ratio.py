# 2023-12-15 : Run the final version
# 2023-12-08 : Making the combined dataloader with pneu MNIST + synthetic
# 2023-12-10 : Concat real + synthetic. label 쪽 오류 많이 뜸. 
# 2023-12-12 : Concat error 해결

# Run Model 2

# Model : Deit
# Model 1 (Base) : pneumonia MNIST => test with pneumonia MNIST
# Model 2 : pneumonia MNIST + Synthetic => test with pneumonia MNIST
# Model 3 : train pneumonia MNIST => Fine-tune with synthetic => test with pnuemonia MNIST 


# slurm-282171.out : RATIO = 1
# slurm-282172.out : RATIO = 100
# slurm-282174.out : RATIO = 300
# slurm-284306.out : RATIO = 200

"""# Setting"""
import argparse
import numpy as np

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
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import trange
import timm

# Set ratio
RATIO = 300

# Set path
import os
PATH = os.getcwd()
os.chdir('..')
PREPATH = os.getcwd() # for dataset and pretrained model
HOME = '/home/s1/chaieunlee'

# Set seed
import random
random.seed(2023)
torch.manual_seed(2023)

# Check time
import time
from datetime import timedelta
start_time = time.time()

# Check cuda
#torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""# 1.Dataloader"""
BATCH_SIZE = 128

"""## Med MNIST"""
import medmnist
from medmnist import INFO, Evaluator
from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT
print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# Settings
data_flag = 'pneumoniamnist' # Channel=1 (black/white image)
download = True

gpu_ids = '0, 1'
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

        label = np.array(label).astype(int) #make same format with real data
        label = label.reshape(1)
        
        return img, label

# load the data
data_dir = f'{HOME}/Transfer/synthetic/'
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
"""# Model 2) Add Synthetic data to pneumonia train data """
# check time
model_start_time = time.time()

#model_name = f'DeiT_pneu_synthetic_{RATIO}'
model_name = 'DeiT_pneu_synthetic_'+ str(RATIO)
print(f'==> Train {model_name}...')

# Load backbone model
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=False)
# 구조 똑같이 바꾸고
model.head = nn.Linear(192, 14)
model.head_dist = nn.Linear(192, 14) #14 classes

# Load chest MNIST pretrained 
model.load_state_dict(torch.load('/home/s1/chaieunlee/Transfer/pretrain_chest/DeiT_chest_pretrained.pth'))

# class pneumonia로
model.head = nn.Linear(192, 2)
model.head_dist = nn.Linear(192, 2)

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
#        print(lbl.size())
        lbl = torch.tensor(lbl, dtype=torch.long)
#        lbl = lbl.type(torch.LongTensor)
#        lbl = lbl.type(torch.FloatTensor)
        img, lbl = img.to(device), lbl.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output[0], lbl.squeeze(dim=-1)) # input:output[0], target: lbl
#        print("Check shape of loss ==>", output[0].shape, lbl.squeeze(dim=-1).shape)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        _, pred = output[0].max(dim=1)

        corr += pred.eq(lbl.squeeze(dim=-1)).sum().item()
#        print("Check shape ==>", pred.shape, lbl.squeeze(dim=-1).shape)
#        print(corr)
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
          lbl = torch.tensor(lbl, dtype=torch.long)
          #lbl = lbl.type(torch.LongTensor)
          img, lbl = img.to(device), lbl.to(device)

          output = model(img) #(batch, num_classes)
          _, pred = output.max(dim=1)

          corr += torch.sum(pred.eq(lbl.squeeze(dim=-1))).item()
#          print("Check shape ==>", pred.shape, lbl.squeeze(dim=-1).shape)
#          print(corr)
          running_loss += loss_fn(output,lbl.squeeze(dim=-1)).item() * img.size(0)

        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc

print("========= Train Model =========")
min_loss = np.inf
total_val_loss, total_val_acc = 0,0
for epoch in range(0, NUM_EPOCHS, 1): # evaluate every epoch
    train_loss, train_acc = model_train(model, train_dev_loader, loss, optimizer, device)
    val_loss, val_acc = model_evaluate(model, val_dev_loader, loss, device)

    scheduler.step()

    # Avg용 loss, acc += 코드 넣기
    total_val_loss += val_loss
    total_val_acc += val_acc

    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(model.state_dict(), f'{PATH}/{model_name}.pth')

    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
print("-----------------------------")
total_val_loss = total_val_loss/NUM_EPOCHS
total_val_acc = total_val_acc/NUM_EPOCHS
print(f'Avg_val_loss: {total_val_loss:.5f}, Avg_val_acc: {total_val_acc:.5f}')

model_end_time = time.time()
print(str(timedelta(seconds=(model_end_time - model_start_time))), " takes for training ", model_name, "!")


"""# 3. Evaluation """
print('====== Evaluation by checkpoint======')
""" # 1) Deit Base """
# If stopped, bring it from checkpoint
# model 불러오기
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=False)
# 구조 똑같이 바꾸고
model.head = nn.Linear(192, 2)
model.head_dist = nn.Linear(192, 2) #14 classes
# pth 불러오기
model.load_state_dict(torch.load(f'{PATH}/DeiT_pneu_synthetic_{RATIO}.pth'))
model.to(device)
#model.eval()

def model_evaluate(model, data_loader, loss_fn, device):

    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        prograss_bar = tqdm(data_loader)

        for img, lbl in prograss_bar:
          lbl = torch.tensor(lbl, dtype=torch.long)
          #lbl = lbl.type(torch.LongTensor)
          img, lbl = img.to(device), lbl.to(device)

          output = model(img) #(batch, num_classes)
          _, pred = output.max(dim=1)

          corr += torch.sum(pred.eq(lbl.squeeze(dim=-1))).item()
#          print("Check shape ==> ", pred.shape,  lbl.squeeze(dim=-1).shape)
#          print(corr)
          running_loss += loss_fn(output,lbl.squeeze(dim=-1)).item() * img.size(0)

        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc

# model
# test_loader
loss = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val_loss, val_acc = model_evaluate(model, test_loader, loss, device)
print(f'test_loss: {val_loss:.5f}, test_accuracy: {val_acc:.5f}')
