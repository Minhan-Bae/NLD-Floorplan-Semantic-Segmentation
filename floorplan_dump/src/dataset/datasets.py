import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import resize as resize_function

class FloorplanDataset(Dataset):
    def __init__(self, args, type='train'):
        self.img_dir = args.img_dir
        self.img_size = args.image_size
        self.offset = args.image_offset
        with open(args.annotation_file) as file:
            self.data = json.load(file)
        
        self.transform = transforms.Compose([
            transforms.ToTensor()  # 이미지를 Tensor로 변환합니다.
        ])

    def __getitem__(self, idx):
        img_name = list(self.data.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name + '.png')
        img = Image.open(img_path)
        orig_size = np.array(img.size)  # 원본 이미지의 크기를 저장합니다.

        annotation = self.data[img_name]
        min_x = min_y = np.inf
        max_x = max_y = -np.inf
        for key, values in annotation.items():
            instance_coords = []  
            for value in values:
                coords = [float(coord) for coord in value.split(',')]
                coords = np.array(coords).reshape(-1, 2)
                min_x = min(min_x, coords[:, 0].min())
                min_y = min(min_y, coords[:, 1].min())
                max_x = max(max_x, coords[:, 0].max())
                max_y = max(max_y, coords[:, 1].max())
                instance_coords.append(coords)
            annotation[key] = instance_coords

        # offset을 추가하여 crop 범위를 계산합니다.
        min_x = max(0, min_x - self.offset)
        min_y = max(0, min_y - self.offset)
        max_x = min(orig_size[0], max_x + self.offset)
        max_y = min(orig_size[1], max_y + self.offset)

        # 이미지를 crop합니다.
        img = img.crop((min_x, min_y, max_x, max_y))

        # 좌표도 동일하게 조정합니다.
        for key, instance_coords in annotation.items():
            adjusted_coords = []  
            for coords in instance_coords:
                coords -= np.array([min_x, min_y])  # crop에 따른 좌표 조정
                coords = (coords / np.array([max_x - min_x, max_y - min_y])) * np.array(self.img_size)  # Resize에 따른 좌표 조정
                adjusted_coords.append(coords)
            annotation[key] = adjusted_coords

        img = resize_function(img, (self.img_size[1], self.img_size[0]))  # 이미지 크기를 조정합니다.
        img = self.transform(img)

        return img, annotation

    def __len__(self):
        return len(self.data)
