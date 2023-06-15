#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from utils.importmod import *

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
gc.collect()


def main(configs):
    args = arg_parser.parse_args(configs)
    seed_utils.seed_everything(args.seed)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # initialize logger
    log = logger.get_logger(args.log_dir)
    for arg in vars(args):
        s = f"{arg}: {getattr(args, arg)}"
        logging.info(s) 
    
    # initial learning configuration
    
    # dataset & dataloader
    trainLoader, validLoader = Dataloader(args)
    
    for epoch in range(args.epochs):
        msg = f'{epoch}\t{len(trainLoader)}\t{len(validLoader)}'
        logging.info(msg)


if __name__=='__main__':
    config = C
    
    main(config)
    