import os
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegDataset(Dataset):
    def __init__(self, args, type='train', transform=None):
        super().__init__()
        self.type = type
        self.transform = transform
        self.image_dir = args.image_dir
        self.data_json = args.data_json
        
    def __len__(self):
        return len(self.data_json)
        
    
        
    