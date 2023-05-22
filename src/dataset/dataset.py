import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FloorplanDataset(Dataset):
    def __init__(self, img_dir, json_path, img_size=(208, 284)):
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transforms.Compose([transforms.Resize(img_size)])
        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = list(self.data.keys())[idx]  # Assuming the key is the image name
        img_path = os.path.join(self.img_dir, img_name+'.png')
        img = Image.open(img_path)

        # Resizing image
        orig_size = np.array(img.size)
        img = self.transform(img)
        scale = np.array(img.size) / orig_size

        annotation = self.data[img_name]

        # annotation에서 각 항목에 대해 좌표 문자열을 2차원 배열로 변환
        for key, value in annotation.items():
            # 각 문자열을 ','로 분리하고, 숫자형으로 변환하여 2차원 배열로 재구성
            coords = [val.split(',') for val in value]
            coords = [float(c) for sublist in coords for c in sublist]
            coords = np.array(coords).reshape(-1, 2)

            # Scale coordinates
            coords = coords * scale

            annotation[key] = coords.tolist()  # 다시 리스트로 변환하여 저장

        return img, annotation