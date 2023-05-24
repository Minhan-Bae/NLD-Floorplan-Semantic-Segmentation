# import modules and packages
import os
import torch
from datetime import datetime

# logging
ROOT = "/root/workspace/nld_floorplan_seg"

DATE = datetime.now().strftime("%Y-%m-%d")
TIME = datetime.now().strftime("%H-%M")

DEVICE = ','.join([str(i) for i in range(torch.cuda.device_count())])
MODEL = ''

EPOCH = 100
LR = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 6e-4
WARM_UP = 25

SEED = 2023
BATCH_SIZE = 64 * len(DEVICE) # number of gpu * 64(T4*2)
WORKERS = 4 * len(DEVICE) # number of gpu * 4

EARLY_STOP_CNT = 50
VALID_TERM = 5

IMG_SIZE = (284, 208)
IMG_OFFSET = 10