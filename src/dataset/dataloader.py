import torch
from torch.utils import data

from .augmentations import get_augmentation
from .datasets import FloorPlanDataset

# def collate_fn(batch):
#     # 데이터 형식 유효성 검사
#     lengths = [len(s) for s in batch]
#     if not all(length == lengths[0] for length in lengths):
#         raise ValueError("All samples should have the same length.")

#     # 배치로 묶인 데이터 생성
#     transposed_batch = list(zip(*batch))  # 튜플로 묶인 배치
#     data = torch.stack(transposed_batch[0], dim=0)  # 데이터 특성

#     # 레이블 처리
#     labels = transposed_batch[1][0]  # 리스트의 첫 번째 요소 추출
#     labels = torch.tensor(labels, dtype=torch.float32)  # 레이블을 텐서로 변환

#     # 특성별로 묶인 튜플 형태로 데이터 반환
#     return (data, labels)


def Dataloader(args):
    dataset = FloorPlanDataset(args, transform=get_augmentation(data_type="train"))
    # dataset = FloorPlanDataset(args)
    # train_ratio = 0.8
    # train_size = int(len(dataset) * train_ratio)
    # valid_size = len(dataset) - train_size
    
    valid_size = 256
    train_size = len(dataset) - valid_size
    dataset_train, dataset_valid = data.random_split(dataset, [train_size, valid_size])
    
    
    train_loader = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True, sampler=None, pin_memory=True
    )
    valid_loader = data.DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True, sampler=None, pin_memory=True
    )
    
    # train_loader = data.DataLoader(
    #     dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True, collate_fn=collate_fn, sampler=None, pin_memory=True
    # )
    # valid_loader = data.DataLoader(
    #     dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True, collate_fn=collate_fn, sampler=None, pin_memory=True
    # )
    
    return train_loader, valid_loader
    