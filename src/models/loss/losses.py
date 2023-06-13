import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()

	def forward(self, inputs, targets, smooth=1):
		num_classes = inputs.size(1)
		true_hot = torch.eye(num_classes)[targets].cuda()

		# [B,H,W,C] -> [B,C,H,W]
		true_hot = true_hot.permute(0, 3, 1, 2).float()
		probas = F.softmax(inputs, dim=1)
		true_hot = true_hot.type(inputs.type())
		dims = (0,) + tuple(range(2, targets.ndimension()))
		# TP
		intersection = torch.sum(probas * true_hot, dims)
		# TP + FP + FN + TN
		cardinality = torch.sum(probas + true_hot, dims)
		dice = ((2. * intersection + smooth) / (cardinality + smooth)).mean()

		return 1 - dice


class FocalLoss(nn.Module):
	def __init__(self, gamma=2, alpha=.25, eps=1e-7, weights=None):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.eps = eps
		self.weight = weights

	def forward(self, inputs, targets):
		logp = F.log_softmax(inputs, dim=1)
		ce_loss = F.nll_loss(logp, targets, weight=self.weight, reduction='none')
		pt = torch.exp(-ce_loss)
		loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

		return loss.mean()


class IoULoss(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(IoULoss, self).__init__()

	def forward(self, inputs, targets, smooth=1):
		inputs = F.sigmoid(inputs)
		inputs = inputs.view(-1)
		targets = targets.view(-1)

		intersection = (inputs * targets).sum()
		total = (inputs + targets).sum()
		union = total - intersection
		IoU = (intersection + smooth) / (union + smooth)

		return 1 - IoU


# Dice + CE
class DiceCELoss(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(DiceCELoss, self).__init__()

	def forward(self, inputs, targets, gamma=0.8, smooth=1):
		num_classes = inputs.size(1)
		true_hot = torch.eye(num_classes)[targets]

		true_hot = true_hot.permute(0, 3, 1, 2).float()
		probas = F.softmax(inputs, dim=1)
		true_hot = true_hot.type(inputs.type())
		dims = (0,) + tuple(range(2, targets.ndimension()))
		intersection = torch.sum(probas * true_hot, dims)
		cardinality = torch.sum(probas + true_hot, dims)
		dice_loss = (2. * intersection / (cardinality + 1e-7)).mean()
		dice_loss = (1 - dice_loss)

		ce = F.cross_entropy(inputs, targets, reduction='mean')
		dice_ce = ce * gamma + dice_loss * (1 - gamma)

		return dice_ce


# Dice + Focal
class DiceFocalLoss(nn.Module):
	def __init__(self, alpha = 0.75):
		super(DiceFocalLoss, self).__init__()
		self.alpha = alpha
		self.dice = DiceLoss()
		self.focal = FocalLoss()

	def forward(self, inputs, targets, eps = 1e-8):
		dice_loss = self.dice(inputs , targets)
		focal_loss = self.focal(inputs , targets)
		dice_focal_loss = focal_loss * self.alpha + dice_loss * (1-self.alpha)

		return dice_focal_loss