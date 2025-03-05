import warnings
warnings.filterwarnings("ignore")
import argparse

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

## loss, optimizer, scheduler 를    위한 라이브러리
from pytorch_toolbelt import losses as PL
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from madgrad import MADGRAD

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

## 이미지 시각화를 위한 라이브러리
from PIL import Image
import webcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from utils.visualize_instance_segmentation import visualize_instance_segmentation
from utils import *

from datasets import *
from configs import *
from models import *
sns.set()
plt.rcParams["axes.grid"] = False

