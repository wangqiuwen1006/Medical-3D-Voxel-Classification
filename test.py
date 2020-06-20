# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:59:05 2020

@author: Apple
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
from Dense3D import Dense3D
from resnet3d import resnet26,resnet50,resnet101,resnet152
from densenet import densenet121,densenet169,densenet201
from vgg import vgg11_bn,vgg13_bn,vgg16_bn,vgg19_bn
from acsconv.converters.acsconv_converter import ACSConverter
from senet import seresnet50
from googlenet import googlenet
from shake_shake import shake_resnet26_2x32d
import csv

test_path='./test/test/'

real_size = 100
crop_size = 32
model_num=5


with open('sampleSubmission.csv','rt') as csvfile:
    reader_test_name = csv.DictReader(csvfile)
    column_test_name = [row['name'] for row in reader_test_name]

test_num = len(column_test_name)
test_data = np.ones((test_num,1,crop_size,crop_size,crop_size))

for i in range(test_num):
    precandidate = np.load(test_path + column_test_name[i] + '.npz')
    test_data[i,0,:,:,:] = precandidate['voxel'][round((real_size-crop_size)/2):round((real_size+crop_size)/2),\
              round((real_size-crop_size)/2):round((real_size+crop_size)/2),round((real_size-crop_size)/2):round((real_size+crop_size)/2)]*\
        precandidate['seg'][round((real_size-crop_size)/2):round((real_size+crop_size)/2),\
              round((real_size-crop_size)/2):round((real_size+crop_size)/2),round((real_size-crop_size)/2):round((real_size+crop_size)/2)]
        
test_loader = data.DataLoader(test_data, batch_size=test_num, shuffle=False, drop_last=True)


model=resnet26(2)
model.load_state_dict(torch.load('netres26.pkl'))
model=model.cuda()
with torch.no_grad():
    for i in range(test_num):
        for step, test_x in enumerate(test_loader):
        # TODO:forward + backward + optimize
        #test_x = torch.tensor(test_x, dtype=torch.float32)
            test_x = test_x.float()
            test_x = test_x.cuda()
            prediction1 = model(test_x)
model=model.cpu()

model=resnet50(2)
model.load_state_dict(torch.load('netmixres50.pkl'))
model=model.cuda()

with torch.no_grad():
    for i in range(test_num):
        for step, test_x in enumerate(test_loader):
        # TODO:forward + backward + optimize
        #test_x = torch.tensor(test_x, dtype=torch.float32)
            test_x = test_x.float()
            test_x = test_x.cuda()
            prediction2 = model(test_x)
model=model.cpu()
        
model=densenet169()
model = ACSConverter(model)
model.load_state_dict(torch.load('netden169.pkl'))
model=model.cuda()

with torch.no_grad():
    for i in range(test_num):
        for step, test_x in enumerate(test_loader):
        # TODO:forward + backward + optimize
        #test_x = torch.tensor(test_x, dtype=torch.float32)
            test_x = test_x.float()
            test_x = test_x.cuda()
            prediction3 = model(test_x)
model=model.cpu()
        
model=vgg16_bn()
model = ACSConverter(model)
model.load_state_dict(torch.load('netvgg16.pkl'))
model=model.cuda()

with torch.no_grad():
    for i in range(test_num):
        for step, test_x in enumerate(test_loader):
        # TODO:forward + backward + optimize
        #test_x = torch.tensor(test_x, dtype=torch.float32)
            test_x = test_x.float()
            test_x = test_x.cuda()
            prediction4 = model(test_x)
model=model.cpu()        

        
model=shake_resnet26_2x32d(2)
model = ACSConverter(model)
model.load_state_dict(torch.load('netshake.pkl'))
model=model.cuda()

with torch.no_grad():
    for i in range(test_num):
        for step, test_x in enumerate(test_loader):
        # TODO:forward + backward + optimize
        #test_x = torch.tensor(test_x, dtype=torch.float32)
            test_x = test_x.float()
            test_x = test_x.cuda()
            prediction5 = model(test_x)
model=model.cpu()

prediction1=prediction1.cpu()
prediction2=prediction2.cpu()
prediction3=prediction3.cpu()
prediction4=prediction4.cpu()
prediction5=prediction5.cpu()
result1=np.array(prediction1)[:,1]
result2=np.array(prediction2)[:,1]
result3=np.array(prediction3)[:,1]
result4=np.array(prediction4)[:,1]
result5=np.array(prediction5)[:,1]

answer=np.zeros((test_num))

for i in range(test_num):
    note=np.zeros((model_num))
    if float(result1[i])>=0.5:
        note[0]=1
    if float(result2[i])>=0.5:
        note[1]=1
    if float(result3[i])>=0.5:
        note[2]=1
    if float(result4[i])>=0.5:
        note[3]=1
    if float(result5[i])>=0.5:
        note[4]=1
    if (note.sum()/model_num)>=0.5:
        answer[i]=1
test_name = np.array(column_test_name).reshape(test_num)
test_dict = {'name':test_name, 'predicted':answer}
result = pd.DataFrame(test_dict, index = [0 for _ in range(test_num)])
result.to_csv("submission.csv", index = False, sep = ',')

    
