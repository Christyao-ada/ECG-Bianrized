import torch
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def get_data(classCount, split_ratio, random_seed):
	classes = torch.load(os.path.join('../data', str(classCount), 'classes.pt'))
	x = torch.load(os.path.join('../data', str(classCount), 'source_x.pt'))
	y = torch.load(os.path.join('../data', str(classCount), 'source_y.pt'))
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=random_seed)
	return (classes, x_train, x_test, y_train, y_test)


class TrainDatasets(Dataset):
	def __init__(self, x_train, y_train):
		self.len = x_train.size(0)
		self.x_train = x_train
		self.y_train = y_train

	def __getitem__(self, index):
		return self.x_train[index], self.y_train[index]

	def __len__(self):
		return self.len


class TestDatasets(Dataset):
	def __init__(self, x_test, y_test):
		self.len = x_test.size(0)
		self.x_test = x_test
		self.y_test = y_test

	def __getitem__(self, index):
		return self.x_test[index], self.y_test[index]

	def __len__(self):
		return self.len


class Loader():
	def __init__(self, classCount=5, split_ratio=0.2, random_seed=32, batch_size=20):
		self.classes, self.x_train, self.x_test, self.y_train, self.y_test = get_data(classCount, split_ratio, random_seed)
		self.batch_size = batch_size
		self.train_dataset = TrainDatasets(self.x_train, self.y_train)
		self.test_dataset = TestDatasets(self.x_test, self.y_test)

	def loader(self):
		train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
		return (self.classes, train_loader, test_loader)
