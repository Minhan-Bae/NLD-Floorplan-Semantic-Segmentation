import os
import gc
import warnings
import time
import logging
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from models.model_test import DeepLabV3Plus
from configs import default as C
from dataset.dataloader import Dataloader
from utils import arg_parser, seed_utils, logger
from utils.early_stopping import EarlyStopping
from utils.average_meter import AverageMeter
from utils.visualize_instance_segmentation import visualize_instance_segmentation