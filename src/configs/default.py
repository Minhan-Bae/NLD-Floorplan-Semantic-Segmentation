# import modules and packages
import os
import torch
from datetime import datetime

# logging
ROOT = "/root/workspace/nld_floorplan_seg"
DATE = datetime.now().strftime("%Y-%m-%d")
TIME = datetime.now().strftime("%H-%M")

# environment parameter
DEVICE = ','.join([str(i) for i in range(torch.cuda.device_count())])
BATCH_SIZE = 4 * len(DEVICE) # number of gpu * 64(T4*2)
WORKERS = 4 * len(DEVICE) # number of gpu * 4
SEED = 2023
EPOCH = 100

# model parameter
MODEL = ""
LOSS = "BELOSS"
OPTIM = "AdamW"

# hyperparameter
LR = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 6e-4
WARM_UP = 25

# validation control parameter
EARLY_STOP_CNT = 50
MILE_STONE = "15,25,30"
VALID_TERM = 5

# data configuration parameter
DATA_ROOT = "/mnt/a/dataset"
IMAGE_DIR = os.path.join(DATA_ROOT, "image")
JSON_PATH = os.path.join(DATA_ROOT, "merged.json")

# IMG_SIZE = (284, 208) # raw size(923* 676)
IMG_SIZE = (512, 512)
IMG_OFFSET = 10 # default offset