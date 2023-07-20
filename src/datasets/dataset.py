import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class FloorPlanDataset(Dataset):
    def __init__(self, args, type='train', transform=None):
        super().__init__()
        self.type = type
        self.transform = transform
        self.csvPath = args.csv_path
        self.imageResize = args.image_resize
        self.numClasses = args.n_classes  # 클래스 개수
        self.padValue = args.padding

        with open(self.csvPath, 'r') as f:
            self.dataList = [line.strip().split(',') for line in f.readlines()]
        
    def _generate_label(self, value):
        label = np.zeros((self.numClasses, self.imageResize[0], self.imageResize[1]), dtype=np.float32)
        for i in range(self.numClasses): # background, others 삭제
            label[i] = (value == i+1) * 255
        return label


    def __len__(self):
        return len(self.dataList)


    def __getitem__(self, index):
        _, imagePath, labelPath = self.dataList[index]

        image = Image.open(imagePath).convert('L')
        label = Image.open(labelPath).convert('L')
        
        image.point(lambda x: 0 if x < 110 else 255, '1') #binary
        image = TF.resize(image, (self.imageResize[1], self.imageResize[0]))
        
        label = TF.resize(label, (self.imageResize[1], self.imageResize[0]), interpolation=Image.NEAREST)

        image = np.array(image)
        label = np.array(label)

        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        image = torch.tensor(image, dtype=torch.float)# .permute(2, 0, 1)
        image = image.unsqueeze(0) # if 1ch
        label = torch.tensor(self._generate_label(label), dtype=torch.float)
            
        image = (image - image.min()) / 255
        label = (label - label.min()) / 255
        return image, label