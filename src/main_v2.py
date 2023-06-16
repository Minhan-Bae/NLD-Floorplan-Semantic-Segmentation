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
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_loss = 0.0

        savePath = os.path.join(
                    f"{args.log_dir}",
                    f"image_logs",
                    f'epoch({str(epoch).zfill(len(str(C.EPOCH)))}).png')
        
        end = time.time()
        pbar = tqdm(enumerate(trainLoader), total=len(trainLoader))
        for idx, (features, labels) in pbar:
            features.cuda(non_blocking=True)
            labels.cuda(non_blocking=True)
            
            batch_time.update(time.time() - end)
            msg = (
                "Epoch: {}\t".format(str(epoch).zfill(len(str(args.epochs))))
                + "LR: {:.8f}\t".format(args.base_lr)
                + "Time: {:.3f} ({:.3f})\t".format(batch_time.val, batch_time.avg)
                + "Loss: {:.8f}\t".format(train_loss / len(trainLoader))
            )
        
        visualize_instance_segmentation(
            features[:16].cpu(),
            labels[:16].cpu(),
            shape=(4, 4),
            size=16,
            save=savePath
        )
        logging.info(msg)


if __name__=='__main__':
    config = C
    
    main(config)
    