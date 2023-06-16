import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation(data_type):
    if data_type == 'train':
        return A.Compose(
            [
                A.Rotate(limit=90, border_mode=0, p=0.5),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.MedianBlur(),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                ToTensorV2()
            ]
        )