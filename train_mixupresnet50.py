# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import numpy as np
#import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from torch.autograd import Variable
import time
import pandas as pd
from vgg import vgg16_bn
from resnet3d import resnet26,resnet50,resnet101,resnet152
from acsconv.converters.acsconv_converter import ACSConverter



NUM_EPOCHS = 100
real_size = 100
crop_size = 32
BATCH_SIZE = 32


train_path='./train_val/train_val/'
test_path='./test/test/'
import csv
with open('train_val.csv','rt') as csvfile:
    reader_train_name = csv.DictReader(csvfile)
    column_train_name = [row['name'] for row in reader_train_name]
with open('sampleSubmission.csv','rt') as csvfile:
    reader_test_name = csv.DictReader(csvfile)
    column_test_name = [row['name'] for row in reader_test_name]
with open('train_val.csv','r') as csvfile:
    reader_train_label = csv.DictReader(csvfile)
    column_train_label = [row['label'] for row in reader_train_label]

train_num = round(len(column_train_name)*0.9)
val_num=len(column_train_name)-train_num
test_num = len(column_test_name)
train_data = np.ones((train_num,1,crop_size,crop_size,crop_size))
val_data = np.ones((val_num,1,crop_size,crop_size,crop_size))
test_data = np.ones((test_num,1,crop_size,crop_size,crop_size))
train_label=np.ones((train_num))
val_label=np.ones((val_num))

for i in range(train_num):
    precandidate = np.load(train_path + column_train_name[i] + '.npz')
    train_data[i,0,:,:,:] = precandidate['voxel'][round((real_size-crop_size)/2):round((real_size+crop_size)/2),\
              round((real_size-crop_size)/2):round((real_size+crop_size)/2),round((real_size-crop_size)/2):round((real_size+crop_size)/2)]*\
    precandidate['seg'][round((real_size-crop_size)/2):round((real_size+crop_size)/2),\
                round((real_size-crop_size)/2):round((real_size+crop_size)/2),round((real_size-crop_size)/2):round((real_size+crop_size)/2)]
    train_label[i]=column_train_label[i]
for i in range(val_num):
    precandidate = np.load(train_path + column_train_name[i+train_num] + '.npz')
    val_data[i,0,:,:,:] = precandidate['voxel'][round((real_size-crop_size)/2):round((real_size+crop_size)/2),\
              round((real_size-crop_size)/2):round((real_size+crop_size)/2),round((real_size-crop_size)/2):round((real_size+crop_size)/2)]*\
    precandidate['seg'][round((real_size-crop_size)/2):round((real_size+crop_size)/2),\
                round((real_size-crop_size)/2):round((real_size+crop_size)/2),round((real_size-crop_size)/2):round((real_size+crop_size)/2)]
    val_label[i]=column_train_label[i+train_num]
for i in range(test_num):
    precandidate = np.load(test_path + column_test_name[i] + '.npz')
    test_data[i,0,:,:,:] = precandidate['voxel'][round((real_size-crop_size)/2):round((real_size+crop_size)/2),\
              round((real_size-crop_size)/2):round((real_size+crop_size)/2),round((real_size-crop_size)/2):round((real_size+crop_size)/2)]*\
        precandidate['seg'][round((real_size-crop_size)/2):round((real_size+crop_size)/2),\
              round((real_size-crop_size)/2):round((real_size+crop_size)/2),round((real_size-crop_size)/2):round((real_size+crop_size)/2)]

train_data=torch.from_numpy(train_data)
val_data=torch.from_numpy(val_data)
column_train_label=list(map(int,column_train_label))
train_label=torch.Tensor(train_label)
val_label=torch.Tensor(val_label)

BATCH_SIZE=32

torch_dataset_train = data.TensorDataset(train_data,train_label)
torch_dataset_val=data.TensorDataset(val_data,val_label)
train_loader = data.DataLoader(dataset=torch_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=torch_dataset_val, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(test_data, batch_size=test_num, shuffle=False)

def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



model = resnet50(2)
model = ACSConverter(model)
model=model.cuda()
# TODO:define loss function and optimiter
criterion = torch.nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(), lr=0.0001)

mixup=True
mixup_alpha=0.4

# train and evaluate
for epoch in range(NUM_EPOCHS):
    print(epoch)
    train_loss = 0
    train_acc = 0
    val_loss=0
    val_acc=0
    total=0
    correct=0

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs = inputs.float()
        targets = targets.long()
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, mixup_alpha, 'cuda')
            
            outputs = model(inputs)
            
            loss = mixup_criterion(
                criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weight
        optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if mixup:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()

    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total
    print("[Epoch: %d] Train Loss: %5.5f Train Accuracy: %5.5f" % (epoch+1, train_loss, train_acc))
    with torch.no_grad():
        for step,(batch_x,batch_y) in enumerate(val_loader):

            batch_x = batch_x.float()
            batch_x = batch_x.cuda()
            out = model(batch_x)           #正向传播
            batch_y = batch_y.long()
            batch_y=batch_y.cuda()
            lossvalue = criterion(out,batch_y)        #求损失值
       
            val_loss+=lossvalue
            _,pred = out.max(1)
            num_correct = (pred == batch_y).sum()
            acc = int(num_correct) / val_data.shape[0]
            val_acc += acc
            batch_x=batch_x.cpu()
            batch_y=batch_y.cpu()
        val_loss=val_loss.cpu()
        val_loss=val_loss.data.numpy()/val_num

        print("[Epoch: %d] Val Loss: %5.5f Val Accuracy: %5.5f" % (epoch+1, val_loss, val_acc))

model.cpu()
torch.save(model.state_dict(),'netmixres50.pkl')      

model.eval()

model.load_state_dict(torch.load('netmixres50.pkl'))
model=model.cuda()



for i in range(test_num):
    for step, test_x in enumerate(test_loader):

        test_x = test_x.float()
        test_x = test_x.cuda()
        prediction = model(test_x)

test_dict = {'name':column_test_name, 'predicted':prediction}
prediction = prediction.cpu()
prediction = prediction.data.numpy()
result = pd.DataFrame(test_dict, index = [0 for _ in range(test_num)])
result.to_csv("resultmixres50.csv", index = False, sep = ',')

 