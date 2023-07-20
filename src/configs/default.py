import os
import torch
from datetime import datetime


SEED = 00
EPOCHS = 100

VALID_TERM = 5
EARLY_STOP_CNT = 5
MILE_STONE = "15,25,30"

LR = 1e-6
MOMENTUM = 0.9
WEIGHT_DECAY = 6e-4
WARM_UP = 25

RESIZE_FACTOR = (256, 256)

PADDING = 20

# logging
ROOT = "/mnt/a/workspace/repository/nld-floorplan-segmentation"
DATE = datetime.now().strftime("%Y-%m-%d")
TIME = datetime.now().strftime("%H-%M")
EXP_NUM = f"{DATE}-{TIME}"

CSV_PATH = os.path.join(ROOT, "src/datasets/dataset.csv")
DEVICE = [str(n) for n in range(torch.cuda.device_count())]
BATCH_SIZE = {
    "TRAIN": len(DEVICE) * 64,
    "VALID": 64
}
WORKERS = len(DEVICE) * 4

N_CLASSES = 7
MODEL = {
    "ARCHITECTURE": "UNetPlusPlus",
    "ENCODER": "resnet34",
    "WEIGHTS": "imagenet",
    "CHANNEL": 1, 
    "RESUME": None
}
LOSS = "BalancedBCEWithLogitsLoss"
OPTIM = "MADGRAD"

RESUME = None
