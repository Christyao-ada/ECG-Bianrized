import torch
import torch.nn as nn

from torch.nn import functional as F


def bin_wei(x, w):
	# element = x / w
	# le_clip = element.lt(-1.0).type(torch.float32)
	# ri_clip = element.ge(1.0).type(torch.float32)
	# clip_l = torch.bitwise_and(element.ge(-1.0), element.lt(0.0))
	# clip_r = torch.bitwise_and(element.ge(0.0), element.lt(1.0))
	# cliped = clip_l * (2 + element) * element + clip_r * (2 - element) * element
	# out = cliped + ri_clip - le_clip
	out = torch.tanh(x / w)

	return out


def bin_act(x):
	bin_act = torch.sign(x).detach()
	out = torch.tanh(x)
	return bin_act + out - out.detach()


class BinActivation(nn.Module):
	def __init__(self):
		super(BinActivation, self).__init__()
		self.sign = bin_act

	def forward(self, x):
		out = x
		return out


class BinConv1d(nn.Conv1d):
	def __init__(self, *kargs, **kwargs):
		super(BinConv1d, self).__init__(*kargs, **kwargs)
		self.w = 1
		self.sign = bin_wei

	def set_mode(self, w):
		self.w = w

	def get_weight(self):
		return self.sign(self.weight)

	def forward(self, x):
		return F.conv1d(x, self.sign(self.weight, self.w), stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Linear):
	def __init__(self, *kargs, **kwargs):
		super(BinaryLinear, self).__init__(*kargs, **kwargs)
		self.w = 1
		self.sign = bin_wei

	def set_mode(self, w):
		self.w = w

	def get_weight(self):
		return self.sign(self.weight)

	def forward(self, x):
		return F.linear(x, self.sign(self.weight, self.w))


class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


class BiNet(nn.Module):
	def __init__(self, classCount):
		super(BiNet, self).__init__()

		self.wei_w = 1

		self.conv01 = BinConv1d(1, 8, 16, stride=2, padding=7, bias=False)
		self.conv02 = BinConv1d(8, 12, 12, stride=2, padding=5, bias=False)
		self.conv03 = BinConv1d(12, 32, 9, stride=1, padding=4, bias=False)
		self.conv04 = BinConv1d(32, 64, 7, stride=1, padding=3, bias=False)
		self.conv05 = BinConv1d(64, 64, 5, stride=1, padding=2, bias=False)
		self.conv06 = BinConv1d(64, 64, 3, stride=1, padding=1, bias=False)
		self.conv07 = BinConv1d(64, 72, 3, stride=1, padding=1, bias=False)
		self.linear = BinaryLinear(216, classCount, bias=False)
		self.pool01 = nn.MaxPool1d(8, stride=4)
		self.pool02 = nn.MaxPool1d(4, stride=2)
		self.pool03 = nn.MaxPool1d(5, stride=2)
		self.pool04 = nn.MaxPool1d(4, stride=2)
		self.pool05 = nn.MaxPool1d(2, stride=2)
		self.pool06 = nn.MaxPool1d(2, stride=2)
		self.pool07 = nn.AvgPool1d(2, stride=2)
		self.batn01 = nn.BatchNorm1d(8)
		self.batn02 = nn.BatchNorm1d(12)
		self.batn03 = nn.BatchNorm1d(32)
		self.batn04 = nn.BatchNorm1d(64)
		self.batn05 = nn.BatchNorm1d(64)
		self.batn06 = nn.BatchNorm1d(64)
		self.batn07 = nn.BatchNorm1d(72)
		self.actv01 = BinActivation()
		self.actv02 = BinActivation()
		self.actv03 = BinActivation()
		self.actv04 = BinActivation()
		self.actv05 = BinActivation()
		self.actv06 = BinActivation()
		self.actv07 = BinActivation()
		self.flatten = Flatten()

	def set_wei_mode(self, w):
		self.wei_w = w
		self.conv01.set_mode(self.wei_w)
		self.conv02.set_mode(self.wei_w)
		self.conv03.set_mode(self.wei_w)
		self.conv04.set_mode(self.wei_w)
		self.conv05.set_mode(self.wei_w)
		self.conv06.set_mode(self.wei_w)
		self.conv07.set_mode(self.wei_w)
		self.linear.set_mode(self.wei_w)

	def get_wei_mode(self):
		return (self.wei_w)

	def get_weight(self):
		return (
			self.conv01.get_weight(),
			self.conv02.get_weight(),
			self.conv03.get_weight(),
			self.conv04.get_weight(),
			self.conv05.get_weight(),
			self.conv06.get_weight(),
			self.conv07.get_weight(),
			self.linear.get_weight()
		)

	def forward(self, x):
		x = self.actv01(self.batn01(self.pool01(self.conv01(x))))
		x = self.actv02(self.batn02(self.pool02(self.conv02(x))))
		x = self.actv03(self.batn03(self.pool03(self.conv03(x))))
		x = self.actv04(self.batn04(self.pool04(self.conv04(x))))
		x = self.actv05(self.batn05(self.pool05(self.conv05(x))))
		x = self.actv06(self.batn06(self.pool06(self.conv06(x))))
		x = self.actv07(self.batn07(self.pool07(self.conv07(x))))
		x = self.linear(self.flatten(x))
		return x
