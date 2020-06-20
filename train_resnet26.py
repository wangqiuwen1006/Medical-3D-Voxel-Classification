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
from resnet3d import resnet26,resnet50,resnet101,resnet152
from densenet import densenet121,densenet169,densenet201
from vgg import vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn
from acsconv.converters.acsconv_converter import ACSConverter
from shake_shake import shake_resnet26_2x32d



NUM_EPOCHS = 100
real_size = 100
crop_size = 32


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
train_loader = data.DataLoader(dataset=torch_dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = data.DataLoader(dataset=torch_dataset_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = data.DataLoader(test_data, batch_size=test_num, shuffle=False, drop_last=True)


model = resnet26(2)
model=model.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(), lr=0.0001)

# train and evaluate
for epoch in range(NUM_EPOCHS):
    print(epoch)
    train_loss = 0
    train_acc = 0
    val_loss=0
    val_acc=0
    for step,(batch_x,batch_y) in enumerate(train_loader):

        optimizer.zero_grad()       #优化器梯度归零
        batch_x = batch_x.float()
        batch_x = batch_x.cuda()
        out = model(batch_x)           #正向传播
        batch_y = batch_y.long().cuda()
        lossvalue = criterion(out,batch_y)        #求损失值
        optimizer.zero_grad()       #优化器梯度归零
        lossvalue.backward()    #反向转播，刷新梯度值
        optimizer.step()        #优化器运行一步，注意optimizer搜集的是model的参数
        
        train_loss+=lossvalue
        _,pred = out.max(1)
        num_correct = (pred == batch_y).sum()
        acc = int(num_correct) / train_data.shape[0]
        train_acc += acc
        
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
torch.save(model.state_dict(),'netres26.pkl')

model.load_state_dict(torch.load('netres26.pkl'))
model=model.cuda()
for i in range(test_num):
    for step, test_x in enumerate(test_loader):
        test_x = test_x.float()
        test_x = test_x.cuda()
        prediction = model(test_x)

prediction = prediction.cpu()
prediction = prediction.data.numpy()
test_dict = {'name':column_test_name, 'predictied':prediction}

result = pd.DataFrame(test_dict, index = [0 for _ in range(test_num)])
result.to_csv("resultres26.csv", index = False, sep = ',')

    

    