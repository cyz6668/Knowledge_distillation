import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from Mydata import Mydataset
from Mynet2 import ResNet18
import torch.nn.functional as F



train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        # transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=240),
        transforms.ToTensor(),
        transforms.Normalize([123.675,116.28,103.53],[0.017,0.017,0.017])
])

batch_size = 32
num_classes =5

dir_name="/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/non_vehicle_multiclass/test_images"

labels_file='/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/non_vehicle_multiclass/test_label.txt'

test_dataset=Mydataset(label_file=labels_file,picture_dir=dir_name,transform=train_transforms)
test_data =DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
test_data_size = len(test_dataset)



def func():
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    device="cpu"
    # model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/trained_models/KD_T_student_model_25_91.020526349655750.6951182147253204.pth',map_location = 'cpu') 94
    # model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_ori_11_84.02_2.63.pth') 87
    # model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_T_student_model_11_89.21_0.75.pth',map_location = 'cpu') #89
    # model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_T_online_student_model_10_85.10_1.79.pth')   #90
    
    
    # model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_ori_15_84.44_2.54.pth')   69.59%
    # model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_T_student_model_45_91.75_0.67.pth')  #94.63%
    # model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_T_online_student_model_29_86.75_1.68.pth') #91.99%
    # model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_feature_student_model_19_63.60_9.89.pth') #82.05%
    model=torch.load('/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_feature_online_student_model_15_63.18_9.89.pth')
    test_acc = 0
    total=0
    model=model.to(device)
    for i, data in enumerate(test_data):
        inputs = data['image'].to(device)
        labels = data["label"].to(device)
        total += labels.size(0)

        outputs,stu_f = model(inputs)
        out_result = torch.softmax(outputs, dim=1)
        out_result=out_result.reshape(-1,5)
        _,predictions=out_result.max(1)
        test_acc += predictions.eq(labels).sum().item()
    print(test_acc/total)
    print(str(test_acc)+" "+str(total))

func()