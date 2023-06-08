import torch.nn as nn
import segmentation_models_pytorch as smp


# DeepLabV3
class DeepLabV3(nn.Module):
	def __init__(self, num_classes= 7):
		super(DeepLabV3, self).__init__()
		self.model = smp.DeepLabV3(
				encoder_name='efficientnet-b3',
				encoder_weights='imagenet',
				in_channels=3, classes=num_classes
				)
	def forward(self, x):
		return self.model(x)


#DeepLabV3Plus
class DeepLabV3Plus(nn.Module):
	def __init__(self, num_classes= 7):
		super(DeepLabV3Plus, self).__init__()
		self.model = smp.DeepLabV3Plus(
				encoder_name='efficientnet-b3',
				encoder_weights='imagenet',
				in_channels=3, classes=num_classes
				)
	def forward(self, x):
		return self.model(x)


# UnetPlusPlus
class UnetPlusPlus(nn.Module):
	def __init__(self, num_classes= 7):
		super(UnetPlusPlus, self).__init__()
		self.model = smp.UnetPlusPlus(
				encoder_name='efficientnet-b3',
				encoder_weights='imagenet',
				in_channels=3, classes=num_classes
				)
	def forward(self, x):
		return self.model(x)