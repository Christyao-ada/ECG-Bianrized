import torch
import torch.nn as nn

import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.nn import functional as F


def bin_act(x):
	bin_act = torch.sign(x).detach()
	le_clip = x.lt(-1.0).type(torch.float32)
	ri_clip = x.ge(1.0).type(torch.float32)
	clip_l = torch.bitwise_and(x.ge(-1.0), x.lt(0.0))
	clip_r = torch.bitwise_and(x.ge(0.0), x.lt(1.0))
	cliped = clip_l * (2 + x) * x + clip_r * (2 - x) * x
	out = cliped + ri_clip - le_clip
	# out = torch.tanh(x)
	return bin_act + out - out.detach()


def bin_wei(x, scalar):
	bin_wei = (scalar * torch.sign(x)).detach()
	out = torch.tanh(x)
	return bin_wei + out - out.detach()


class BinActivation(nn.Module):
	def __init__(self):
		super(BinActivation, self).__init__()

	def forward(self, x):
		out = bin_act(x)
		return out


class Flatten(nn.Module):
	def forward(self, x):
		batch_size = x.shape[0]
		return x.view(batch_size, -1)


class BinConv1d(nn.Module):
	def __init__(
		self,
		in_channels,
		out_channels,
		kernel_size=3,
		stride=1,
		padding=0
	):
		super(BinConv1d, self).__init__()
		self.stride = stride
		self.padding = padding
		self.weights_cnt = in_channels * out_channels * kernel_size
		self.shape = (out_channels, in_channels, kernel_size)
		self.weights = nn.Parameter(torch.rand((self.weights_cnt, 1)) * 0.32, requires_grad=True)

	def forward(self, x):
		real_weights = self.weights.view(self.shape)
		scaling_fatcor = torch.mean(torch.mean(torch.abs(real_weights), dim=2, keepdim=True), dim=1, keepdim=True)
		scaling_fatcor = scaling_fatcor.detach()
		bin_weight = bin_wei(real_weights, scaling_fatcor)

		return F.conv1d(x, bin_weight, stride=self.stride, padding=self.padding)


class BinLinear(nn.Module):
	def __init__(
		self,
		in_features,
		out_features
	):
		super(BinLinear, self).__init__()
		self.weight_cnt = in_features * out_features
		self.shape = (out_features,  in_features)
		self.weights = nn.Parameter(torch.rand((self.weight_cnt, 1)) * 0.32, requires_grad=True)

	def forward(self, x):
		real_weights = self.weights.view(self.shape)
		scaling_fatcor = torch.mean(torch.abs(real_weights), dim=1, keepdim=True)
		scaling_fatcor = scaling_fatcor.detach()
		bin_weight = bin_wei(real_weights, scaling_fatcor)

		return F.linear(x, bin_weight)


class BiNet(nn.Module):
	def __init__(self, classCount):
		super(BiNet, self).__init__()
		self.cnn = nn.Sequential(
			BinConv1d(1, 8, 16, stride=2, padding=7),
			nn.MaxPool1d(kernel_size=8, stride=4),
			nn.BatchNorm1d(8),
			BinActivation(),

			BinConv1d(8, 12, 12, padding=5, stride=2),
			nn.MaxPool1d(4, stride=2),
			nn.BatchNorm1d(12),
			BinActivation(),

			BinConv1d(12, 32, 9, stride=1, padding=4),
			nn.MaxPool1d(5, stride=2),
			nn.BatchNorm1d(32),
			BinActivation(),

			BinConv1d(32, 64, 7, stride=1, padding=3),
			nn.MaxPool1d(4, stride=2),
			nn.BatchNorm1d(64),
			BinActivation(),

			BinConv1d(64, 64, 5, stride=1, padding=2),
			nn.MaxPool1d(2, 2),
			nn.BatchNorm1d(64),
			BinActivation(),

			BinConv1d(64, 64, 3, stride=1, padding=1),
			nn.MaxPool1d(2, 2),
			nn.BatchNorm1d(64),
			BinActivation(),

			BinConv1d(64, 72, 3, stride=1, padding=1),
			nn.MaxPool1d(2, 2),
			nn.BatchNorm1d(72),
			BinActivation(),

			Flatten(),
			BinLinear(in_features=216, out_features=classCount)
		)

	def forward(self, x, ex_features=None):
		return self.cnn(x)
