import albumentations as A

# augmentation
def get_augmentation(data_type):
    if data_type == 'train':
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightness(),
                        A.RandomGamma(),
                        A.GaussianBlur(),
                        A.MotionBlur(),
                        # A.ColorJitter(),
                        # A.ToSepia()                                            
                    ]
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]
        )