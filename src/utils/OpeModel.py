import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam


class OpeModel():
	def __init__(self, model, device, lr, trLoader, teLoader):
		self.device = device
		self.model = model.to(device)
		self.train_loader = trLoader
		self.test_loader = teLoader
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = Adam(self.model.parameters(), lr=lr)
		self.acc_list, self.loss_list = [], []

	def modify_lr(self, value):
		for para in self.optimizer.param_groups:
			para['lr'] = value

	def train(self):
		self.model.train()
		for data in self.train_loader:
			inputs, target = data
			inputs = inputs.to(self.device)
			target = target.to(self.device)
			self.optimizer.zero_grad()
			outputs = self.model(inputs)
			loss = self.criterion(outputs, target)
			loss.backward()
			self.optimizer.step()
		loss_item = loss.cpu().item()
		return loss_item

	def test(self):
		self.model.eval()
		correct, total = 0, 0
		for data in self.test_loader:
			inputs, target = data
			inputs = inputs.to(self.device)
			target = target.to(self.device)
			outputs = self.model(inputs)
			_, predicted = torch.max(outputs.data, dim=1)
			total += len(target)
			correct += (predicted == target).sum().cpu().item()
		correct_item = (100 * correct) / total
		return correct_item

	def train_strategy_c(self, epoch):
		for index in range(epoch):
			loss = self.train()
			acc = self.test()
			self.loss_list.append(loss)
			self.acc_list.append(acc)
			print('Loss, Acc - %d: %.8f, %.2f %%' % (index, loss, acc))

	def train_strategy_a(self, w, epoch):
		self.model.set_wei_mode(w)
		for index in range(epoch):
			loss = self.train()
			acc = self.test()
			print('Loss, Acc - %d: %.8f, %.2f %%' % (index, loss, acc))

	def eval_acc(self):
		self.model.eval()
		correct, total = 0, 0
		for data in self.test_loader:
			inputs, target = data
			inputs = inputs.to(self.device)
			target = target.to(self.device)
			outputs = self.model(inputs)
			_, predicted = torch.max(outputs.data, dim=1)
			total += len(target)
			correct += (predicted == target).sum().cpu().item()
		acc = (100 * correct) / total
		print('model acc: %.2f' % (acc))
		return acc

	def save_state(self, mode, end, cnt=5):
		acc = self.eval_acc()
		torch.save(self.model, '../model/%s,%.2f,%d,%d.pt' % (mode, acc, end, cnt))
		np.save('../tmp/%s,acc,%d,%d' % (mode, end, cnt), self.acc_list)
		np.save('../tmp/%s,loss,%d,%d' % (mode, end, cnt), self.loss_list)

	def load_state(self, mode, acc, end, cnt):
		self.model = torch.load('../model/%s,%.2f,%d,%d.pt' % (mode, acc, end, cnt), map_location=self.device)
		self.acc_list = np.load('../tmp/%s,acc,%d,%d.npy' % (mode, end, cnt)).tolist()
		self.loss_list = np.load('../tmp/%s,loss,%d,%d.npy' % (mode, end, cnt)).tolist()

	def get_state(self):
		return (self.model, self.acc_list, self.loss_list)
