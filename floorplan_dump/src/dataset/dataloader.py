from .augmentations import get_augmentation
from .datasets import FloorplanDataset
from torch.utils import data

import sys
sys.path.append('..')
def Dataloader(args):
    dataset_train = FloorplanDataset(
        
    )

