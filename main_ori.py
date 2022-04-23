from sklearn.metrics import average_precision_score
import timm
from PIL import Image
import cv2
import lmdb
import pickle
from skimage import io, transform
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
class betweenLoss(nn.Module):
    def __init__(self, gamma=[1, 1, 1, 1, 1, 1], loss=nn.L1Loss()):
        super(betweenLoss, self).__init__()
        self.gamma = gamma
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        length = len(outputs)
        res = sum([self.gamma[i] * self.loss(outputs[i], targets[i]) for i in range(length)])
        return res




train_transforms = transforms.Compose(
        [transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),#随机裁剪到256*256
        # transforms.RandomRotation(degrees=15),#随机旋转
        transforms.RandomHorizontalFlip(),#随机水平翻转
        transforms.CenterCrop(size=240),#中心裁剪到224*224
        transforms.ToTensor(),#转化成张量
        transforms.Normalize([123.675,116.28,103.53],[0.017,0.017,0.017])
])

test_valid_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225])])

batch_size = 16
num_classes =5

# train_dataset = Mydataset(label_file="non_vehicle_multiclass/training_label.txt",picture_dir='non_vehicle_multiclass/images/',transform=train_transforms)
# train_dataset = Mydataset(label_file="non_vehicle_multiclass/training_label2.txt",picture_dir='non_vehicle_multiclass/for_multicls/',transform=train_transforms)
# train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
# train_data_size = len(train_datasets)
train_dataset=Mydataset(label_file="/mnt/cfs/users/yuezi.chen/d-cv-pytorch/training_label_balanced.txt",picture_dir='/mnt/cfs/users/yuezi.chen/d-cv-pytorch/images_cls_balanced',transform=train_transforms)
train_data =DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_data_size = len(train_dataset)


student=ResNet18()
# for param in student.parameters():
#     param.requires_grad = False


loss_func = nn.NLLLoss()
optimizer = optim.Adam(student.parameters())

def cross_entropy_loss(output, target):
	return -torch.sum(output * target) / output.shape[0]

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()


criterion = [nn.CrossEntropyLoss(), cross_entropy_loss]


update_parameters = [{'params': student.parameters()}]
optimizer = optim.SGD(update_parameters, lr=0.1, momentum=0.9, weight_decay=5e-4)  # nesterov = True, weight_decay = 1e-4，stage = 3, batch_size = 64
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150 , 250],gamma=0.1)



def train_and_valid(model, loss_function, optimizer, epochs=25):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")#若有gpu可用则用gpu
    record = []
    best_acc = 0.0
    best_epoch = 0
    data_rec=[]
    model=model.to(device)
    for epoch in range(epochs):#训练epochs轮
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        correct=0
        model.train()#训练
        train_loss = 0.0
        train_acc = 0
        total=0
        scheduler.step()
        for i, data in enumerate(train_data):
            inputs = data['image'].to(device)
            labels = data["label"].to(device)
            total += labels.size(0)
            optimizer.zero_grad()

            outputs,stu_f = model(inputs)
            out_result = torch.softmax(outputs, dim=1)
            out_result=out_result.reshape(-1,5)
     
            loss_lambda=0.5
            temperature=3
            

            out_result=outputs.reshape(-1,5)
            loss=F.cross_entropy(out_result,labels)
            loss.backward()

            optimizer.step()
            
            train_loss += loss.item() 

            #_,predictions=outputs.max(1)
            _,predictions=out_result.max(1)
            train_acc += predictions.eq(labels).sum().item()
            print(str(i)+"/"+str(len(train_data))+"  G_loss:"+str(train_loss / (i + 1))+"  acc:"+str( 100. * train_acc / total)+"  | "+str(train_acc)+"/"+str(total))
            
            
           
        avg_train_loss = train_loss / total
        avg_train_acc = train_acc / total

        record.append([avg_train_loss,  avg_train_acc,])

       

        epoch_end = time.time()

        tmp_str="Epoch:"+str(epoch+1)+"Training: Loss:"+str(avg_train_loss)+"Accuracy: "+str(avg_train_acc)+" \n"
        data_rec.append(tmp_str)
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n Time: {:.4f}s".format(
                epoch + 1,avg_train_loss, avg_train_acc * 100, 
                epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))      

        torch.save(model, '/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/KD_ori_' + str(epoch + 1)+ '_'+str(format(avg_train_acc * 100,'.2f'))+'_'+str(format(avg_train_loss * 100,'.2f'))+ '.pth')
    f=open("/mnt/cfs/users/yuezi.chen/pytorch-retinanet-master/KD/rec_ori.txt",'a')
    f.write("\n")
    lens=len(data_rec)
    for i in range(lens):
        f.write(data_rec[i])
    f.close()
    return model, record


if __name__=='__main__':
    num_epochs = 50
    trained_model, record = train_and_valid(student, loss_func, optimizer, num_epochs)
  
  
