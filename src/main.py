#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import warnings

warnings.filterwarnings("ignore")

import gc

gc.collect()

import argparse
import time
import logging
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.cuda.empty_cache()

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from configs import default
from utils import (
    arg_parser, seed_utils, trainer
)

if __name__=='__main__':
    # initial step
    C = default
    seed_utils.seed_everything(C.SEED)

    # parse argument
    args = arg_parser.parse_args(C)
    arg_parser.print_args(args)
    
    print(type(args))
    print(args.criterion)
    # torch.cuda.set_device(args.device_id[0])
    
    early_cnt = 0
    for epoch in range(1, args.epochs):
        scaler = GradScaler()
        
        # adjust learning rate
        lr = adjust_learning_rate()
    
    