import os
import sys
import logging
import argparse

# from str_to_bool import *

from ..configs import default as C

def parse_args(C):
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Facial Landmark Detection")

    # Define data paths
    parser.add_argument("--image-dir", type=str)  # Path to the image directory
    parser.add_argument("--train-csv-path", type=str)  # Path to the training data CSV file
    parser.add_argument("--valid-csv-path", type=str)  # Path to the validation data CSV file

    parser.add_argument("--log-dir", default=f"{C.ROOT}/logs/{C.DATE}/{C.TIME}", type=str)  # Log directory path

    parser.add_argument("--gpus", default=C.DEVICE, type=str)  # GPU device(s) to use
    parser.add_argument("--epochs", default=C.EPOCH, type=int)  # Number of epochs
    parser.add_argument("--batch-size", default=C.BATCH_SIZE, type=int)  # Batch size
    parser.add_argument("--base-lr", default=C.LR, type=float)  # Base learning rate
    parser.add_argument("--workers", default=C.WORKERS, type=int)  # Number of workers for data loading

    parser.add_argument("--criterion", default="wingloss", type=str)  # Loss function selection
    parser.add_argument("--optimizer", default="SGD", type=str)  # Optimization algorithm selection
    parser.add_argument("--momentum", default=C.MOMENTUM, type=float)  # Momentum value for SGD
    parser.add_argument("--weight-decay", default=C.WEIGHT_DECAY, type=float)  # Weight decay
    parser.add_argument("--valid-term", default=C.VALID_TERM, type=int)  # Validation interval
    parser.add_argument("--early-stop", default=C.EARLY_STOP_CNT, type=int)  # Early stopping condition
    parser.add_argument("--seed", default=C.SEED, type=int)  # Random seed
    parser.add_argument("--milestones", default="15,25,30", type=str)  # Learning rate decay milestones
    parser.add_argument("--warmup", default=C.WARM_UP, type=int)  # Warm-up epochs
    parser.add_argument("--resume", default=None, type=str)  # Path to the saved model for resuming training
    parser.add_argument("--valid-initial", default="true", type=str)  # Perform initial validation or not

    # Parse the arguments
    args = parser.parse_args()

    # Perform additional operations (optional)
    args.devices_id = [int(d) for d in args.gpus.split(",")]  # Convert GPU device(s) to a list of integers
    args.milestones = [int(m) for m in args.milestones.split(",")]  # Convert learning rate decay milestones to a list

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Set log and save directories
    os.makedirs(os.path.join(args.log_dir, "image_logs"), exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "model_logs"), exist_ok=True)

    return args

def print_args(args):
    for arg in vars(args):
        s = arg + ": " + str(getattr(args, arg))
        logging.info(s)

# Test parse_args
if __name__ == "__main__":
    args = parse_args(C)

    # Print parsed arguments
    print_args(args)