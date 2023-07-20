import os
import logging
import argparse

def parse_args(C):
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Floorplan Segmentation")

    # Define data paths
    parser.add_argument("--csv-path", default=C.CSV_PATH, type=str)
    parser.add_argument("--log-dir", default=f"{C.ROOT}/logs/{C.DATE}/{C.TIME}", type=str)  # Log directory path
    parser.add_argument("--image-resize", default=C.RESIZE_FACTOR)

    parser.add_argument("--device_id", default=C.DEVICE, type=list)  # GPU device(s) to use
    parser.add_argument("--epochs", default=C.EPOCHS, type=int)  # Number of epochs
    parser.add_argument("--train-batch", default=C.BATCH_SIZE["TRAIN"], type=int)  # Batch size
    parser.add_argument("--valid-batch", default=C.BATCH_SIZE["VALID"], type=int)
    
    parser.add_argument("--base-lr", default=C.LR, type=float)  # Base learning rate
    parser.add_argument("--workers", default=C.WORKERS, type=int)  # Number of workers for data loading

    parser.add_argument("--criterion", default=C.LOSS, type=str)  # Loss function selection
    parser.add_argument("--optimizer", default=C.OPTIM, type=str)  # Optimization algorithm selection
    parser.add_argument("--momentum", default=C.MOMENTUM, type=float)  # Momentum value for SGD
    parser.add_argument("--weight-decay", default=C.WEIGHT_DECAY, type=float)  # Weight decay
    parser.add_argument("--valid-term", default=C.VALID_TERM, type=int)  # Validation interval
    parser.add_argument("--early-stop", default=C.EARLY_STOP_CNT, type=int)  # Early stopping condition
    parser.add_argument("--seed", default=C.SEED, type=int)  # Random seed
    parser.add_argument("--milestones", default=C.MILE_STONE, type=str)  # Learning rate decay milestones
    parser.add_argument("--warmup", default=C.WARM_UP, type=int)  # Warm-up epochs
    parser.add_argument("--valid-initial", default="true", type=str)  # Perform initial validation or not

    parser.add_argument("--n-classes", default=C.N_CLASSES, type=int)
    parser.add_argument("--architecture", default=C.MODEL["ARCHITECTURE"], type=str)
    parser.add_argument("--encoder", default=C.MODEL["ENCODER"], type=str)
    parser.add_argument("--weights", default=C.MODEL["WEIGHTS"], type=str)
    parser.add_argument("--in-channels", default=C.MODEL["CHANNEL"], type=str)
    parser.add_argument("--resume", default=C.MODEL["RESUME"], type=str)  # Path to the saved model for resuming training
    parser.add_argument("--padding", default=C.PADDING, type=int)
    # Parse the arguments
    args = parser.parse_args()

    # Perform additional operations (optional)
    args.gpus = ','.join(C.DEVICE)
    args.milestones = [int(m) for m in args.milestones.split(",")]  # Convert learning rate decay milestones to a list

    # Set log and save directories
    os.makedirs(os.path.join(args.log_dir, "image_logs"), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "model_logs"), exist_ok=True)

    return args