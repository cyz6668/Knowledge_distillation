import numpy as np
import torch
import pandas as pd
import os
from PIL import Image
from skimage import io, transform
class Mydataset(torch.utils.data.Dataset):
    def __init__(self,label_file,picture_dir,transform=None):
        
        self.label=pd.read_table(label_file,sep='\t',names=['name','label'])
        self.image_dir=picture_dir
        self.transform=transform
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
        img_name=os.path.join(self.image_dir,self.label.iloc[idx,0])
        image=Image.open(img_name).convert("RGB")
        label=self.label.iloc[idx,1]
        if self.transform:
            image=self.transform(image)
        image = np.array(image).astype('float32')  # 转换成数组类型浮点型32位
          #读出来的图像是rgb,rgb,rbg..., 转置为 rrr...,ggg...,bbb...
        # image = image/255.0
        # image = transform.resize(image,(256,128))
        # image = image.transpose((2, 0, 1))   
        sample = {'image':image,'label':label}
        # if self.transform:
        #     sample=self.transform(sample['image'])
        return sample

# train_dataset = Mydataset(label_file="non_vehicle_multiclass/training_label.txt",picture_dir='non_vehicle_multiclass/images/')

# for i in range(5):
#     sample=train_dataset[i]
#     # print(sample['image'])
#     print(i,sample['image'].shape,sample['label'])