import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class FloorPlanDataset(Dataset):
    def __init__(self, args, type='train', transform=None):
        # self.imageDirPath = os.path.join(args.data_dir, 'image')
        # self.labelDirPath = os.path.join(args.data_dir, 'label')
        
        # self.imageDir = sorted(os.listdir(self.imageDirPath))
        # self.labelDir = sorted(os.listdir(self.labelDirPath))
        
        # self.dataList = list()
        # for imagePath, labelPath in zip(self.imageDir, self.labelDir):
        #     imagePath = os.path.join(self.imageDirPath, imagePath)
        #     labelPath = os.path.join(self.labelDirPath, labelPath)
        #     self.dataList.append((imagePath, labelPath))
        with open(args.csv_path) as f:
            self.dataList = [line.strip().split(',') for line in f.readlines()]

        self.imageResize = args.image_resize # e.g. (128,128)
        self.type = type
        self.transform = transform


    def __len__(self):
        return len(self.dataList)


    def __getitem__(self, index):       
        _, imagePath, labelPath  = self.dataList[index]


        image = Image.open(imagePath).convert('RGB')
        label = Image.open(labelPath).convert('RGB')

        image = F.resize(image, (self.imageResize[1], self.imageResize[0]))
        label = F.resize(label, (self.imageResize[1], self.imageResize[0]))
        
        image = np.array(image)
        label = np.array(label)
        
        if self.transform:
            transformed = self.transform(image=image, label=label)
            image = transformed['image']
            label = transformed['label']

        image = torch.tensor(image, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
    
        image = (image - image.min()/(image.max() - image.min()))
        label = (label - label.min()/(label.max() - label.min()))

        image = (2 * image) -1
        label = (2 * label) -1
                    
        return image, label