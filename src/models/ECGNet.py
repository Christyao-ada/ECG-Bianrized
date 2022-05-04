import torch
import torch.nn as nn


class Flatten(nn.Module):
	def forward(self, x):
		batch_size = x.shape[0]
		return x.view(batch_size, -1)


class BaselineNet(nn.Module):
	def __init__(self, classCount=5):
		super(BaselineNet, self).__init__()
		self.cnn = nn.Sequential(
			nn.Conv1d(1, 8, 16, stride=2, padding=7, bias=False),
			nn.BatchNorm1d(8),
			nn.MaxPool1d(kernel_size=8, stride=4),
			nn.ReLU(),

			nn.Conv1d(8, 12, 12, stride=2, padding=5, bias=False),
			nn.BatchNorm1d(12),
			nn.MaxPool1d(4, stride=2),
			nn.ReLU(),

			nn.Conv1d(12, 32, 9, stride=1, padding=4, bias=False),
			nn.BatchNorm1d(32),
			nn.MaxPool1d(5, stride=2),
			nn.ReLU(),

			nn.Conv1d(32, 64, 7, stride=1, padding=3, bias=False),
			nn.BatchNorm1d(64),
			nn.MaxPool1d(4, stride=2),
			nn.ReLU(),

			nn.Conv1d(64, 64, 5, stride=1, padding=2, bias=False),
			nn.BatchNorm1d(64),
			nn.MaxPool1d(2, 2),
			nn.ReLU(),

			nn.Conv1d(64, 64, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm1d(64),
			nn.MaxPool1d(2, 2),
			nn.ReLU(),

			nn.Conv1d(64, 72, 3, stride=1, padding=1, bias=False),
			nn.BatchNorm1d(72),
			nn.MaxPool1d(2, 2),
			nn.ReLU(),

			Flatten(),
			nn.Dropout(p=0.1),
			nn.Linear(in_features=216, out_features=classCount, bias=False)
		)

	def forward(self, x):
		return self.cnn(x)
