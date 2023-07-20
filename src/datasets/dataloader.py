from torch.utils import data

def FloorPlanDataloader(dataset, args):    
    valid_size = 1024
    train_size = len(dataset) - valid_size
    dataset_train, dataset_valid = data.random_split(dataset, [train_size, valid_size])
    
    train_loader = data.DataLoader(
        dataset_train, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, drop_last=True, sampler=None, pin_memory=True
    )
    valid_loader = data.DataLoader(
        dataset_valid, batch_size=args.valid_batch, shuffle=False, num_workers=args.workers, drop_last=True, sampler=None, pin_memory=True
    )
    
    return train_loader, valid_loader