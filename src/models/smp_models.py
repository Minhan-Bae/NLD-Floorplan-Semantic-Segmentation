import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SegmentationModel(nn.Module):
    def __init__(self, args):
        super(SegmentationModel, self).__init__()
        self.architecture_name = args.architecture
        self.encoder_name = args.encoder
        self.weights=args.weights
        self.in_channels = args.in_channels
        self.classes = args.n_classes
        

        if self.architecture_name.lower() == 'deeplabv3':
            self.model = smp.DeepLabV3(
                encoder_name=self.encoder_name,
                encoder_weights=self.weights,
                in_channels=self.in_channels,
                classes=self.classes
            )
        elif self.architecture_name.lower()  == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.encoder_name,
                encoder_weights=self.weights,
                in_channels=self.in_channels,
                classes=self.classes
            )
        elif self.architecture_name.lower()  == 'unetplusplus':
            self.model = smp.UnetPlusPlus(
                encoder_name=self.encoder_name,
                encoder_weights=self.weights,
                in_channels=self.in_channels,
                classes=self.classes
            )
        else:
            raise ValueError("Invalid model_name. Supported values are 'DeepLabV3', 'DeepLabV3Plus', and 'UnetPlusPlus'.")

        # Apply DataParallel here
        if args.device_id is not None:
            self.model = nn.DataParallel(self.model, device_ids=[int(i) for i in args.device_id])
        # Load model weights if required
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        return self.model(x)

