import torch
from torch.utils import data

from .augmentations import get_augmentation
from .datasets import FloorPlanDataset

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels


def Dataloader(args, dataset):
    
    
    # train_ratio = 0.8
    # train_size = int(len(dataset) * train_ratio)
    # valid_size = len(dataset) - train_size
    
    # valid_size = 256
    # train_size = len(dataset) - valid_size
    
    # just for testing
    train_size = 256
    valid_size = len(dataset) - train_size
    
    dataset_train, dataset_valid = data.random_split(dataset, [train_size, valid_size])
    
    train_loader = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, collate_fn=collate_fn
    )
    valid_loader = data.DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True, collate_fn=collate_fn
    )
    
    return train_loader, valid_loader
    