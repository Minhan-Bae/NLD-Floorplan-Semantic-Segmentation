# Import modules and packages
from importmod import *
# Enable cuDNN benchmarking for improved performance (use with caution)
torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache()

warnings.filterwarnings("ignore")

# 확인 부분
print("Pytorch version: {}".format(torch.__version__))
print("GPU: {}".format(torch.cuda.is_available()))

print("Device name: ", torch.cuda.get_device_name(0))
print("Device count: ", torch.cuda.device_count())

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(configs, device=device):
    # load arguments
    args = arg_parser.parse_args(configs)
    seed_utils.seed_everything(args.seed)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # init loggine
    logger.get_logger(args.log_dir)
    for arg in vars(args):
        s = f"{arg}: {getattr(args, arg)}"
        logging.info(s) 
    
    
    # model & optimizer
    model = smp_models.SegmentationModel(args)
    criterion = PL.BalancedBCEWithLogitsLoss()
    optimizer = MADGRAD(params=model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1)
    
    # dataset & dataloader
    datasets = dataset.FloorPlanDataset(args, transform=augmentation.get_augmentation(data_type="train"))
    trainLoader, validLoader = dataloader.FloorPlanDataloader(datasets, args)
    
    # init validate
    val_loss_min, val_miou_min = evaluate.valid_one_epoch(args, 
                                                          validLoader, 
                                                          model,
                                                          criterion,
                                                          device,
                                                          save=os.path.join(args.log_dir,
                                                                            "image_logs",
                                                                            f"epoch({str(0).zfill(len(str(args.epochs)))}).png",
                                                                            )
                                                          )
    msg = f"init valid loss & mIoU: {val_loss_min:.4f}, {val_miou_min:.4f}"
    logging.info(msg)
    
    # train & valid
    earlyStopCnt = 0
    for epoch in range(args.epochs):
        ## train
        msg = evaluate.train_one_epoch(args, epoch, trainLoader, model, criterion, optimizer, scheduler, device)
        logging.info(msg)

        if (epoch & args.valid_term == 0) or (epoch == args.epochs):
            val_loss, val_miou = evaluate.valid_one_epoch(args, 
                                                          validLoader, 
                                                          model,
                                                          criterion,
                                                          device,
                                                          save=os.path.join(args.log_dir,
                                                                            "image_logs",
                                                                            f"epoch({str(epoch+1).zfill(len(str(args.epochs)))}).png",
                                                                            )
                                                          )
            s = f"best loss: {val_loss}    best mIoU: {val_miou}"
            logging.info(s)      

            transform_gif.transform_gif(args.log_dir)
            
            if val_miou_min < val_miou and val_loss_min > val_loss: #
                earlyStopCnt = 0
                torch.save(model.state_dict(), os.path.join(args.log_dir, "checkpoint.pth"))
                logging.info(f'Validation loss updates ({val_loss_min:.6f} --> {val_loss:.6f}).')
                logging.info(f'Validation mIoU updates ({val_miou_min:.4f} --> {val_miou:.6f}.  Saving model ...')
                
                val_miou_min = val_miou
                val_loss_min = val_loss

            else:
                earlyStopCnt += 1
                logging.info(f"Early stopping is now {earlyStopCnt}")
                logging.info(f'Validation loss {val_loss:.6f}).')
                logging.info(f'Validation mIoU {val_miou:.6f}.')
                
                if earlyStopCnt > args.early_stop:
                    logging.exception("Early stopping is activated")
                    break
            
            
                  
if __name__=="__main__":
    main(default)
    
    
    
    logging.info("⭐⭐ GPU is free. ⭐⭐")
