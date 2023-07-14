DEBUG = False

# Import modules and Packages

import warnings
warnings.filterwarnings("ignore")

import os
import random
import gc
import easydict
import glob
import multiprocessing
import copy
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

    ## Transform을 위한 라이브러리
import albumentations as A
from albumentations.pytorch import ToTensorV2

    ## Model을 위한 라이브러리
import timm
import segmentation_models_pytorch as smp

    ## Fold를 위한 라이브러리
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

    ## loss, optimizer, scheduler 를 위한 라이브러리
from pytorch_toolbelt import losses as PL
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from madgrad import MADGRAD

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

    ## # Weight & bias
    ## import wandb

    ## 이미지 시각화를 위한 라이브러리
from PIL import Image
import webcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

sns.set()
plt.rcParams["axes.grid"] = False

print("Pytorch version: {}".format(torch.__version__))
print("GPU: {}".format(torch.cuda.is_available()))

print("Device name: ", torch.cuda.get_device_name(0))
print("Device count: ", torch.cuda.device_count())


# set config
def set_config():
    CFG = {}

    CFG['experiment_number'] = ""
    CFG["seed"] = 23

    CFG["data_root"] = "/mnt/a/workspace/repository/nld_floorplan_seg/src/dataset/dataset.csv"
    CFG["resume"] = None
    CFG["lr"] = 1e-6
    CFG["weight_decay"] = 1e-4

    CFG["batch_size"] = 256
    CFG["valid_term"] = 5
    CFG["n_epoch"] = 50
    CFG["n_Folds"] = 5
    CFG["n_iter"] = 3
    CFG["patience"] = 9999

    CFG["n_class"] = 7
    CFG["resize_factor"] = (256,256)
    CFG["pad_value"] = 20

    CFG["ROOT"] = "/mnt/a/workspace/repository/nld_floorplan_seg"
    CFG["DATE"] = datetime.now().strftime("%Y-%m-%d")
    CFG["TIME"] = datetime.now().strftime("%H-%M")
    
    CFG['experiment_number'] = f'{CFG["DATE"]}-{CFG["TIME"]}'
    return CFG

# logging
def print_args(args):
    not_logging = ('dataList', 'resume')
    for key, value in args.items():
        if key not in not_logging:
            s = key + ": " + str(value)
            logging.info(s)

# utils
def seed_everything(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def collate_fn(batch):
    lengths = [len(s) for s in batch]
    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All samples should have the same length.")

    transposed_batch = list(zip(*batch))  # 튜플로 묶인 배치
    data = torch.stack(transposed_batch[0], dim=0)  # 데이터 특성

    labels = transposed_batch[1][0]  # 리스트의 첫 번째 요소 추출
    labels = torch.tensor(labels, dtype=torch.float32)  # 레이블을 텐서로 변환

    return (data, labels)


# dataset
class FloorPlanDataset(Dataset):
    def __init__(self, CFG, type='train', transform=None):
        super().__init__()
        self.type = type
        self.transform = transform
        self.dataList = CFG['dataList']
        self.imageResize = CFG["resize_factor"]
        self.num_classes = CFG["n_class"]  # 클래스 개수
        self.pad_value = CFG["pad_value"]


    def _generate_label(self, value):
        label = np.zeros((self.num_classes, self.imageResize[0], self.imageResize[1]), dtype=np.float32)
        for i in range(self.num_classes): # background, others 삭제
            label[i] = (value == i+1) * 255
        return label


    def __len__(self):
        return len(self.dataList)


    def __getitem__(self, index):
        _, imagePath, labelPath = self.dataList[index]

        image = Image.open(imagePath).convert('L')
        image.point(lambda x: 0 if x < 110 else 255, '1')
        
        label = Image.open(labelPath).convert('L')
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
    

# augmentation
def get_augmentation(data_type):
    '''
    A simple augmentation function
    '''
    if data_type == 'train':
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightness(),
                        A.RandomGamma(),
                        # A.ColorJitter(),
                        # A.ToSepia()                                            
                    ]
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),

                
            ]
        )

def FloorPlanDataloader(dataset, CFG):    
    valid_size = 1024
    train_size = len(dataset) - valid_size
    dataset_train, dataset_valid = data.random_split(dataset, [train_size, valid_size])
    
    train_loader = data.DataLoader(
        dataset_train, batch_size=CFG['batch_size'], shuffle=True, num_workers=4, drop_last=True, sampler=None, pin_memory=True
    )
    valid_loader = data.DataLoader(
        dataset_valid, batch_size=CFG['batch_size'], shuffle=False, num_workers=4, drop_last=True, sampler=None, pin_memory=True
    )
    
    return train_loader, valid_loader


def train_one_epoch(epoch, model, data_loader, criterion, optimizer, scheduler, device):
    model.train()
    model.to(device)
    cnt = 0
    correct = 0
    scaler = GradScaler()

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (image, label) in pbar:

        image = image.to(device)
        label = label.to(device)

        with autocast(enabled=True):
            model = model.to(device)
            output = model(image)
            
            loss = criterion(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        _, preds = torch.max(output, 1)
        preds = preds.unsqueeze(1)  # 두 번째 차원 추가
        correct += torch.sum(preds == label.data)
        cnt += 1

        description = f"| # Epoch : {epoch + 1} Loss : {(loss.item()):.4f}"
        pbar.set_description(description)

    scheduler.step()
    
    msg = (
    "Epoch: {}\t".format(str(epoch).zfill(len(str(CFG['n_epoch']))))
    + "LR: {:.8f}\t".format(CFG['lr'])
    + "Loss: {:.8f}\t".format(loss / len(data_loader))
    )
    return msg


def compute_miou(predictions, targets):
    batch_size = predictions.size(0)
    num_classes = predictions.size(1)
    miou = 0.0

    for c in range(num_classes):
        intersection = 0.0
        union = 0.0

        for i in range(batch_size):
            pred = (predictions[i, c] > 0.5).int()  # 예측된 클래스 c의 마스크
            target = (targets[i, c] > 0.5).int()  # 실제 클래스 c의 마스크

            intersection += (pred & target).sum().item()
            union += (pred | target).sum().item()

        if union == 0:  # union이 0인 경우 처리
            iou = 0.0
        else:
            iou = intersection / union

        miou += iou

    miou /= num_classes

    return miou



def encode_with_cutoff(arr, value, cutoff_value):

    encoded_arr = np.zeros_like(arr)  # 모든 요소를 0으로 초기화한 배열 생성

    # 일정 값 이후의 인덱스에 대해 one-hot encoding 수행
    indices = np.where(arr >= cutoff_value)
    encoded_arr[indices] = value

    return encoded_arr


def visualize_instance_segmentation(images_list,  masks_list, title=None, save=None):
    fig = plt.figure(figsize=(16,16))
    grid = ImageGrid(fig, 111, nrows_ncols=(4,4), axes_pad=0.08)

    for ax, feature, predict in zip(grid, images_list[:16], masks_list[:16]):
        feature = feature.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()

        feature = np.transpose(feature, (1,2,0))   
        # feature = cv2.cvtColor(feature, cv2.COLOR_BGR2GRAY) # (1, 128, 128)
        
        oneHot = np.zeros(CFG["resize_factor"], dtype=np.uint8)
        for idx in range(CFG["n_class"]):
            predict[idx] = (predict[idx] - predict[idx].min()) / (predict[idx].max() - predict[idx].min()) * 255
            p = encode_with_cutoff(predict[idx], idx/CFG["n_class"]*255, 200)
            p = p.astype(np.uint8)
            oneHot |= p
        
        
        ax.imshow(feature, cmap='gray', alpha=0.3)
        ax.imshow(oneHot, cmap='jet', alpha=0.7)
        ax.axis('off')                    
        if save:
            plt.savefig(save)

    # Display the title if specified.
    if title:
        print(title)
    # Save the figure if specified.

    # Show the plot.
    plt.show()
    plt.close()

def valid_one_epoch(valid_loader, model, device, save=None):
    cum_loss = 0.0
    cum_miou = 0.0  # mIoU를 누적할 변수

    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for features, labels in valid_loader:
            features = features.to(device)
            labels = labels.to(device)

            predicts = model(features).to(device)

            loss = PL.balanced_binary_cross_entropy_with_logits(predicts, labels)

            cum_loss += loss.item()
            
            # mIoU 계산
            miou = compute_miou(predicts, labels)
            cum_miou += miou

    if save:
        visualize_instance_segmentation(
            features,
            predicts,
            save=save,
        )
    
    avg_loss = cum_loss / len(valid_loader)
    avg_miou = cum_miou / len(valid_loader)

    return avg_loss, avg_miou


if __name__=="__main__":
    #config part
    CFG = set_config()
    with open(CFG['data_root']) as f:
        CFG['dataList'] = [line.strip().split(',') for line in f.readlines()]

    save_path = os.path.join(CFG["ROOT"],"logs",CFG["DATE"],CFG["TIME"])
    save_image_path = os.path.join(save_path, 'image_logs')
    os.makedirs(save_image_path, exist_ok=True)
    #logging part
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_path,f"{CFG['DATE']}-{CFG['TIME']}.log"), mode='w'),
            logging.StreamHandler(),
        ],
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"


    print_args(CFG)

    seed_everything(CFG["seed"])

    model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=1,
    classes=CFG["n_class"]
    )
    model = nn.DataParallel(model, device_ids=[0, 1])

    if CFG["resume"]:
        checkpoint = torch.load(
            CFG["resume"], map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint, strict=False)
        
    # loss and optimizer
    criterion = PL.BalancedBCEWithLogitsLoss()
    optimizer = MADGRAD(params=model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG["n_epoch"], T_mult=1)
    
    dataset = FloorPlanDataset(CFG, transform=get_augmentation(data_type='train'))
    trainLoader, validLoader = FloorPlanDataloader(dataset, CFG)
    
    # run
    torch.cuda.empty_cache()
    earlyStopCnt = 0

    # init
    val_loss_min, val_miou_min = valid_one_epoch(
        validLoader, model, device,
        save=os.path.join(
            save_image_path,
            f'epoch({str(0).zfill(len(str(CFG["n_epoch"])))}).png',)        
        )
    logging.info(f"init valid loss & mIoU: {val_loss_min:.4f}, {val_miou_min:.4f}")


    for epoch in range(CFG['n_epoch']):
        # train start
        msg = train_one_epoch(epoch, model, trainLoader, criterion, optimizer, scheduler, device)
        logging.info(msg)
        
        # if (epoch & CFG["valid_term"] == 0) or (epoch == CFG["n_epoch"]):
        val_loss, val_miou = valid_one_epoch(
            validLoader, model, device,
            save=os.path.join(
            save_image_path,
            f'epoch({str(epoch+1).zfill(len(str(CFG["n_epoch"])))}).png')        
        )
        s = f"init loss: {val_loss}    init mIoU: {val_miou}"
        logging.info(s)
        
        if val_miou_min < val_miou and val_loss_min > val_loss: #
            earlyStopCnt = 0
            torch.save(model.state_dict(), os.path.join(save_path, "checkpoint.pth"))
            logging.info(f'Validation loss updates ({val_loss_min:.6f} --> {val_loss:.6f}).')
            logging.info(f'Validation mIoU updates ({val_miou_min:.4f} --> {val_miou:.6f}.  Saving model ...')
            
            val_miou_min = val_miou
            val_loss_min = val_loss

        else:
            earlyStopCnt += 1
            logging.info(f"Early stopping is now {earlyStopCnt}")
            logging.info(f'Validation loss {val_loss:.6f}).')
            logging.info(f'Validation mIoU {val_miou:.6f}.')
            
            if earlyStopCnt > CFG["patience"]:
                logging.exception("Early stopping is activated")
                break
            
    # extract gif
    def transform_gif(dir, frame_duration=1200):
        # GIF 파일의 이름과 프레임 속도를 설정합니다.
        gif_name = os.path.join(dir,"output.gif")

        png_files = sorted(glob.glob(os.path.join(dir, "image_logs", "*.png")))
        # 첫 번째 PNG 파일을 열어 GIF 파일의 기본 설정을 합니다.
        with Image.open(png_files[0]) as first_image:
            # 이미지 크기 조정
            width, height = first_image.size
            resized_images = []

            # 각 PNG 파일을 GIF에 추가하기 위해 크기를 조정합니다.
            for png_file in png_files:
                with Image.open(png_file) as image:
                    resized_image = image.resize((width, height))
                    resized_images.append(resized_image)

            # GIF 파일로 저장합니다.
            resized_images[0].save(
                gif_name,
                format="GIF",
                append_images=resized_images[1:],
                save_all=True,
                duration=frame_duration,
                loop=0,
            )

        print("GIF 파일이 생성되었습니다.")

    transform_gif(save_path)