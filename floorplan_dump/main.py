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

from src.configs import default as C

if __name__=='__main__':
    C