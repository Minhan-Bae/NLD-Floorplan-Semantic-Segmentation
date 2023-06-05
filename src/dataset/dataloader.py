from .augmentations import get_augmentation
from .datasets import FloorplanDataset
from torch.utils import data
from .augmentations import get_augmentation
import sys
sys.path.append('..')
def Dataloader(args):
    dataset_train = FloorplanDataset(
        args,
        type = 'train',
        transform=get_augmentation(data_type="valid")
    )
    dataset_valid = FloorplanDataset(
        args,
        type = 'valid',
        transform=get_augmentation(data_type='valid')
    )
    
    train_loader = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
    )
    valid_loader = data.DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True
    )
    
    return train_loader, valid_loader
    