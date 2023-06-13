#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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
from models import sgd_nan
from configs import default as C
from dataset.dataloader import Dataloader
from utils import arg_parser, seed_utils, logger
from utils import trainer as T
from utils.early_stopping import EarlyStopping

from models.loss import balanced_entropy, class_balanced_loss, losses
from pytorch_toolbelt import losses as L
gc.collect()
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = False
# cudnn.benchmark = True

def main():
    # 초기 설정
    
    # 인자 파싱
    args = arg_parser.parse_args(C)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    seed_utils.seed_everything(args.seed)
    # 로거 설정
    log = logger.get_logger(args.log_dir)
    for arg in vars(args):
        s = arg + ": " + str(getattr(args, arg))
        logging.info(s)

    model = DeepLabV3Plus()
    torch.cuda.set_device(args.devices_id[0])
    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()
    optimizer = sgd_nan.SGD_NanHandler(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    train_loader, valid_loader = Dataloader(args)

    # EarlyStopping 설정
    early_stopping = EarlyStopping(
        patience=args.early_stop,
        verbose=True,
        checkpoint_path='./checkpoint.pt',
        logger=log,
    )

    for epoch in range(1, args.epochs + 1):
        scaler = GradScaler()
        msg = T.train_one_epoch(args, train_loader, model, optimizer, epoch, C.LR, scaler)
        logging.info(msg)


if __name__ == '__main__':
    main()
