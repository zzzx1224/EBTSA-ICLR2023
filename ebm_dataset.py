import numpy as np 
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
# from utils import get_transform
import pdb
import random
import torch
import time
import cv2
import torchvision.transforms as transforms

data_path = '../224kfold/'

class PACS4ebm(Dataset):
	def __init__(self, test_domain, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art_painting', 'photo', 'cartoon', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_train_kfold.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				train_domain_labels.append(int(label)-1)
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			# self.num_imgs.append(len(train_domain_imgs))
		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		seed = 777

		# elif phase == 'val':
		self.domain_list.append(test_domain)
		# pdb.set_trace()
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_crossval_kfold.txt', 'r')
			lines = f.readlines()

			val_domain_imgs = []
			val_domain_labels = []

			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				# self.val_img_list.append(data_path + img)
				# self.val_label_list.append(int(label)-1)
				val_domain_imgs.append(data_path + img)
				val_domain_labels.append(int(label)-1)
			np.random.seed(seed)
			np.random.shuffle(val_domain_imgs)
			np.random.seed(seed)
			np.random.shuffle(val_domain_labels)
			self.val_img_list.append(val_domain_imgs)
			self.val_label_list.append(val_domain_labels)
		self.domain_list.remove(test_domain)


		# else:
		f = open('../files/' + test_domain + '_test_kfold.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label)-1)

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]

			neg_domain = list(range(self.num_domains))
			neg_domain.remove(domain_id)
			# pdb.set_trace()
			self.neg_imgs = []
			self.neg_labels = []
			for neg_domainid in neg_domain:
				self.neg_imgs += self.train_img_list[neg_domainid]
				self.neg_labels += self.train_label_list[neg_domainid]

			if len(self.neg_imgs) < len(self.img_list):
				self.neg_imgs = self.neg_imgs + self.neg_imgs
				self.neg_labels = self.neg_labels + self.neg_labels

			seed = 777
			# seed = np.random.randint(1000)
			np.random.seed(seed)
			np.random.shuffle(self.neg_imgs)
			np.random.seed(seed)
			np.random.shuffle(self.neg_labels)
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		label = self.label_list[item]
		if self.phase == 'train':
			neg_image = Image.open(self.neg_imgs[item]).convert('RGB')  # (C, H, W)
			neg_label = self.neg_labels[item]
			if self.transform is not None:
				neg_image = self.transform(neg_image)
		else:
			neg_label = torch.Tensor(0)
			neg_image = torch.Tensor(0)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		transform_test = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	    ])
		if self.transform is not None:
			# image = transform_test(image)
			image = self.transform(image)

		
		# return image and label
		# return image, self.label_list[item]
		return image, label, neg_image, neg_label

	def __len__(self):
		return len(self.img_list)


class rtPACS(Dataset):
	def __init__(self, test_domain, num_samples=20, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art_painting', 'photo', 'cartoon', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		self.num_samples = num_samples
		assert self.num_domains <= len(self.domain_list)

		self.sample_list = []

		self.infer_imgs = []
		self.infer_labels = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../files/' + self.domain_list[i] + '_train_kfold.txt', 'r')
			lines = f.readlines()
			samples = {}
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				label = int(label) - 1
				if label not in samples.keys():
					samples[label] = []
				samples[label].append(data_path + img)
			self.sample_list.append(samples)

			# pdb.set_trace()

			for i in range(len(samples.keys())):
				self.infer_imgs = self.infer_imgs + samples[i][:num_samples]   # 20 samples for center feature during test
				self.infer_labels = self.infer_labels + [i] * num_samples

		# pdb.set_trace()

	def reset(self, phase, transform=None):
		# pdb.set_trace()
		self.phase = phase
		self.transform = transform
		if phase == 'train':
			self.img_list = []
			self.label_list = []
			for i in range(self.num_domains):
				for j in range(7):
					# pdb.set_trace()
					np.random.shuffle(self.sample_list[i][j])
					self.img_list = self.img_list + self.sample_list[i][j][:self.num_samples]
					self.label_list = self.label_list + [j] * self.num_samples

		else:
			self.img_list = self.infer_imgs
			self.label_list = self.infer_labels

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# print(np.array(image).shape)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)


class Office4ebm(Dataset):
	def __init__(self, test_domain, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art', 'clipart', 'product', 'real_World']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../office-home_files/' + self.domain_list[i] + '_train.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				train_domain_labels.append(int(label))
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []

		seed = 777

		f = open('../office-home_files/' + test_domain + '_train.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label))

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None, neg_transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.neg_transform = neg_transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]

			neg_domain = list(range(self.num_domains))
			neg_domain.remove(domain_id)
			# pdb.set_trace()
			self.neg_imgs = []
			self.neg_labels = []
			for neg_domainid in neg_domain:
				self.neg_imgs += self.train_img_list[neg_domainid]
				self.neg_labels += self.train_label_list[neg_domainid]

			if len(self.neg_imgs) < len(self.img_list):
				self.neg_imgs = self.neg_imgs + self.neg_imgs
				self.neg_labels = self.neg_labels + self.neg_labels

			seed = 777
			# seed = np.random.randint(1000)
			np.random.seed(seed)
			np.random.shuffle(self.neg_imgs)
			np.random.seed(seed)
			np.random.shuffle(self.neg_labels)
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		label = self.label_list[item]
		if self.phase == 'train':
			neg_image = Image.open(self.neg_imgs[item]).convert('RGB')  # (C, H, W)
			neg_label = self.neg_labels[item]
			if self.neg_transform is not None:
				neg_image = self.neg_transform(neg_image)
			elif self.transform is not None:
				neg_image = self.transform(neg_image)
		else:
			neg_label = torch.Tensor(0)
			neg_image = torch.Tensor(0)
		image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		transform_test = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	    ])
		if self.transform is not None:
			# image = transform_test(image)
			image = self.transform(image)

		
		# return image and label
		# return image, self.label_list[item]
		return image, label, neg_image, neg_label

	def __len__(self):
		return len(self.img_list)


class rtOF(Dataset):
	def __init__(self, test_domain, num_samples=20, num_domains=3, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['art', 'clipart', 'product', 'real_World']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		self.num_samples = num_samples
		assert self.num_domains <= len(self.domain_list)

		self.sample_list = []

		self.infer_imgs = []
		self.infer_labels = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../office-home_files/' + self.domain_list[i] + '_train.txt', 'r')
			lines = f.readlines()
			samples = {}
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				label = int(label)
				if label not in samples.keys():
					samples[label] = []
				samples[label].append(data_path + img)
			self.sample_list.append(samples)

			for i in range(len(samples.keys())):
				self.infer_imgs = self.infer_imgs + samples[i][:num_samples]   # 20 samples for center feature during test
				self.infer_labels = self.infer_labels + [i] * num_samples

		# pdb.set_trace()

	def reset(self, phase, domain, transform=None):
		# pdb.set_trace()
		self.phase = phase
		self.transform = transform
		if phase == 'train':
			self.img_list = []
			self.label_list = []
			# for i in range(self.num_domains):
			for j in range(65):
				# pdb.set_trace()
				np.random.shuffle(self.sample_list[domain][j])
				self.img_list = self.img_list + self.sample_list[domain][j][:self.num_samples]
				self.label_list = self.label_list + [j] * self.num_samples

		else:
			self.img_list = self.infer_imgs
			self.label_list = self.infer_labels

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# print(np.array(image).shape)
		image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)

class MNIST4ebm(Dataset):
	def __init__(self, test_domain, train_domains, transform=None):
		# assert phase in ['train', 'val', 'test']
		# self.domain_list = ['0', '15', '30', '45', '60', '75', '90']
		# for te_dom in test_domain:
		# 	self.domain_list.remove(te_dom)
		self.domain_list = train_domains
		# pdb.set_trace()
		# self.num_domains = num_domains
		self.num_domains = len(train_domains)

		self.train_domain_imgs = []
		self.train_domain_labels = []

		data_path = '../mnist/'

		# for i in range(len(self.domain_list)):
		f = open('../mnist/' + self.domain_list[-1] + '_train_imgs.txt', 'r')
		lines = f.readlines()
		all_training_imgs = []
		all_training_labels = []
		
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			all_training_imgs.append(img[2:])
			all_training_labels.append(int(label))

		seed = 777
		np.random.seed(seed)
		np.random.shuffle(all_training_imgs)
		np.random.seed(seed)
		np.random.shuffle(all_training_labels)

		training_imgs = all_training_imgs[:10000]
		training_labels = all_training_labels[:10000]

		# pdb.set_trace()

		for i in range(len(self.domain_list)):
			domain_imgs = []
			domain_labels = []
			for j in range(10000):
				domain_imgs.append(data_path + self.domain_list[i] + training_imgs[j][2:])
				domain_labels.append(training_labels[j])

			self.train_domain_imgs.append(domain_imgs)
			self.train_domain_labels.append(domain_labels)

		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		# elif phase == 'val':
		for i in range(len(self.domain_list)):
			self.domain_imgs = []
			self.domain_labels = []
			f = open('../mnist/' + self.domain_list[i] + '_test_imgs.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.domain_imgs.append(data_path + img)
				self.domain_labels.append(int(label))
				# np.random.seed(seed)
				# np.random.shuffle(domain_imgs)
				# np.random.seed(seed)
				# np.random.shuffle(domain_labels)
			self.val_img_list.append(self.domain_imgs)
			self.val_label_list.append(self.domain_labels)


		# else:
		for i in range(len(test_domain)):
			f = open('../mnist/' + test_domain[i] + '_test_imgs.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.test_img_list.append(data_path + img)
				self.test_label_list.append(int(label))
		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_domain_imgs[domain_id]
			self.label_list = self.train_domain_labels[domain_id]

			neg_domain = list(range(self.num_domains))
			neg_domain.remove(domain_id)
			# pdb.set_trace()
			self.neg_imgs = []
			self.neg_labels = []
			for neg_domainid in neg_domain:
				self.neg_imgs += self.train_domain_imgs[neg_domainid]
				self.neg_labels += self.train_domain_labels[neg_domainid]

			if len(self.neg_imgs) < len(self.img_list):
				self.neg_imgs = self.neg_imgs + self.neg_imgs
				self.neg_labels = self.neg_labels + self.neg_labels

			# seed = 777
			seed = np.random.randint(1000)
			np.random.seed(seed)
			np.random.shuffle(self.neg_imgs)
			np.random.seed(seed)
			np.random.shuffle(self.neg_labels)
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item])#.convert('RGB')  # (C, H, W)
		# image = image.resize((28, 28))
		label = self.label_list[item]
		if self.phase == 'train':
			neg_image = Image.open(self.neg_imgs[item])#.convert('RGB')  # (C, H, W)
			neg_label = self.neg_labels[item]
			if self.transform is not None:
				neg_image = self.transform(neg_image)
		else:
			neg_label = torch.Tensor(0)
			neg_image = torch.Tensor(0)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		transform_test = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	    ])
		if self.transform is not None:
			# image = transform_test(image)
			image = self.transform(image)

		
		# return image and label
		# return image, self.label_list[item]
		return image, label, neg_image, neg_label

	def __len__(self):
		return len(self.img_list)



class Dom4ebm(Dataset):
	def __init__(self, test_domain, num_domains=5, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
		self.domain_list.remove(test_domain)
		self.num_domains = num_domains
		assert self.num_domains <= len(self.domain_list)

		self.train_img_list = []
		self.train_label_list = []

		data_path = '../domainnet/'

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../domainnet/files/' + self.domain_list[i] + '_train.txt', 'r')
			lines = f.readlines()
			train_domain_imgs = []
			train_domain_labels = []
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				train_domain_imgs.append(data_path + img)
				train_domain_labels.append(int(label))
			self.train_img_list.append(train_domain_imgs)
			self.train_label_list.append(train_domain_labels)
			# self.num_imgs.append(len(train_domain_imgs))
		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		seed = 777

		# # elif phase == 'val':
		# self.domain_list.append(test_domain)
		# # pdb.set_trace()
		# for i in range(len(self.domain_list)):
		# 	f = open('../domainnet/files/' + self.domain_list[i] + '_crossval_kfold.txt', 'r')
		# 	lines = f.readlines()

		# 	val_domain_imgs = []
		# 	val_domain_labels = []

		# 	for line in lines:
		# 		[img, label] = line.strip('\n').split(' ')
		# 		# self.val_img_list.append(data_path + img)
		# 		# self.val_label_list.append(int(label)-1)
		# 		val_domain_imgs.append(data_path + img)
		# 		val_domain_labels.append(int(label)-1)
		# 	np.random.seed(seed)
		# 	np.random.shuffle(val_domain_imgs)
		# 	np.random.seed(seed)
		# 	np.random.shuffle(val_domain_labels)
		# 	self.val_img_list.append(val_domain_imgs)
		# 	self.val_label_list.append(val_domain_labels)
		# self.domain_list.remove(test_domain)


		# else:
		f = open('../domainnet/files/' + test_domain + '_test.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label))

		# seed = 777
		# pdb.set_trace()
		np.random.seed(seed)
		np.random.shuffle(self.test_img_list)
		np.random.seed(seed)
		np.random.shuffle(self.test_label_list)

		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_img_list[domain_id]
			self.label_list = self.train_label_list[domain_id]

			neg_domain = list(range(self.num_domains))
			neg_domain.remove(domain_id)
			# pdb.set_trace()
			self.neg_imgs = []
			self.neg_labels = []
			for neg_domainid in neg_domain:
				self.neg_imgs += self.train_img_list[neg_domainid]
				self.neg_labels += self.train_label_list[neg_domainid]

			if len(self.neg_imgs) < len(self.img_list):
				self.neg_imgs = self.neg_imgs + self.neg_imgs
				self.neg_labels = self.neg_labels + self.neg_labels

			seed = 777
			# seed = np.random.randint(1000)
			np.random.seed(seed)
			np.random.shuffle(self.neg_imgs)
			np.random.seed(seed)
			np.random.shuffle(self.neg_labels)
			# pdb.set_trace()

		# elif phase == 'val':
		# 	self.transform = transform
		# 	self.img_list = self.val_img_list[domain_id]
		# 	self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		image = image.resize((224, 224))
		label = self.label_list[item]
		if self.phase == 'train':
			neg_image = Image.open(self.neg_imgs[item]).convert('RGB')  # (C, H, W)
			neg_image = neg_image.resize((224, 224))
			neg_label = self.neg_labels[item]
			if self.transform is not None:
				neg_image = self.transform(neg_image)
		else:
			neg_label = torch.Tensor(0)
			neg_image = torch.Tensor(0)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		transform_test = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	    ])
		if self.transform is not None:
			# image = transform_test(image)
			image = self.transform(image)

		
		# return image and label
		# return image, self.label_list[item]
		return image, label, neg_image, neg_label

	def __len__(self):
		return len(self.img_list)



class FMNIST4ebm(Dataset):
	def __init__(self, test_domain, train_domains, transform=None):
		# assert phase in ['train', 'val', 'test']
		# self.domain_list = ['0', '15', '30', '45', '60', '75', '90']
		# for te_dom in test_domain:
		# 	self.domain_list.remove(te_dom)
		self.domain_list = train_domains
		# pdb.set_trace()
		# self.num_domains = num_domains
		self.num_domains = len(train_domains)

		self.train_domain_imgs = []
		self.train_domain_labels = []

		data_path = '../fashion_MNIST/'

		# for i in range(len(self.domain_list)):
		f = open('../fashion_MNIST/' + self.domain_list[-1] + '_train_imgs.txt', 'r')
		lines = f.readlines()
		all_training_imgs = []
		all_training_labels = []
		
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			all_training_imgs.append(img[2:])
			all_training_labels.append(int(label))

		seed = 777
		np.random.seed(seed)
		np.random.shuffle(all_training_imgs)
		np.random.seed(seed)
		np.random.shuffle(all_training_labels)

		training_imgs = all_training_imgs[:10000]
		training_labels = all_training_labels[:10000]

		# pdb.set_trace()

		for i in range(len(self.domain_list)):
			domain_imgs = []
			domain_labels = []
			for j in range(10000):
				domain_imgs.append(data_path + self.domain_list[i] + training_imgs[j][2:])
				domain_labels.append(training_labels[j])

			self.train_domain_imgs.append(domain_imgs)
			self.train_domain_labels.append(domain_labels)

		# pdb.set_trace()

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []
		# self.transform = transform
		# self.meta_test_domain = np.random.randint(len(self.domain_list))

		# elif phase == 'val':
		for i in range(len(self.domain_list)):
			self.domain_imgs = []
			self.domain_labels = []
			f = open('../fashion_MNIST/' + self.domain_list[i] + '_test_imgs.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.domain_imgs.append(data_path + img)
				self.domain_labels.append(int(label))
				# np.random.seed(seed)
				# np.random.shuffle(domain_imgs)
				# np.random.seed(seed)
				# np.random.shuffle(domain_labels)
			self.val_img_list.append(self.domain_imgs)
			self.val_label_list.append(self.domain_labels)


		# else:
		for i in range(len(test_domain)):
			f = open('../fashion_MNIST/' + test_domain[i] + '_test_imgs.txt', 'r')
			lines = f.readlines()
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				self.test_img_list.append(data_path + img)
				self.test_label_list.append(int(label))
		# pdb.set_trace()

	def reset(self, phase, domain_id, transform=None):
		# pdb.set_trace()
		self.phase = phase
		if phase == 'train':
			self.transform = transform
			self.img_list = self.train_domain_imgs[domain_id]
			self.label_list = self.train_domain_labels[domain_id]

			neg_domain = list(range(self.num_domains))
			neg_domain.remove(domain_id)
			# pdb.set_trace()
			self.neg_imgs = []
			self.neg_labels = []
			for neg_domainid in neg_domain:
				self.neg_imgs += self.train_domain_imgs[neg_domainid]
				self.neg_labels += self.train_domain_labels[neg_domainid]

			if len(self.neg_imgs) < len(self.img_list):
				self.neg_imgs = self.neg_imgs + self.neg_imgs
				self.neg_labels = self.neg_labels + self.neg_labels

			# seed = 777
			seed = np.random.randint(1000)
			np.random.seed(seed)
			np.random.shuffle(self.neg_imgs)
			np.random.seed(seed)
			np.random.shuffle(self.neg_labels)
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list[domain_id]
			self.label_list = self.val_label_list[domain_id]

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item])#.convert('RGB')  # (C, H, W)
		# image = image.resize((28, 28))
		label = self.label_list[item]
		if self.phase == 'train':
			neg_image = Image.open(self.neg_imgs[item])#.convert('RGB')  # (C, H, W)
			neg_label = self.neg_labels[item]
			if self.transform is not None:
				neg_image = self.transform(neg_image)
		else:
			neg_label = torch.Tensor(0)
			neg_image = torch.Tensor(0)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		transform_test = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	    ])
		if self.transform is not None:
			# image = transform_test(image)
			image = self.transform(image)

		
		# return image and label
		# return image, self.label_list[item]
		return image, label, neg_image, neg_label

	def __len__(self):
		return len(self.img_list)


class SVHN4ebm(Dataset):
	def __init__(self, test_domain, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['all']

		self.train_domain_imgs = []
		self.train_domain_labels = []

		data_path = '../mnist/'

		# pdb.set_trace()

		for i in range(len(self.domain_list)):
			f = open('../svhn/' + self.domain_list[i] + '.txt', 'r')
			lines = f.readlines()
			domain_imgs = []
			domain_labels = []
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				domain_imgs.append('../svhn/' + img)
				domain_labels.append(int(label))

			self.train_domain_imgs.append(domain_imgs)
			self.train_domain_labels.append(domain_labels)

		self.val_img_list = []
		self.val_label_list = []
		self.test_img_list = []
		self.test_label_list = []

		# else:
		# for i in range(len(test_domain)):
		f = open('../mnist/' + '0_test_imgs.txt', 'r')
		lines = f.readlines()
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			self.test_img_list.append(data_path + img)
			self.test_label_list.append(int(label))
		# pdb.set_trace()
		self.val_img_list = self.test_img_list
		self.val_label_list = self.test_label_list

	def reset(self, phase, domain_id, transform=None, neg_transform=None):
		self.phase = phase
		# pdb.set_trace()
		if phase == 'train':
			self.transform = transform
			self.neg_transform = neg_transform
			self.img_list = self.train_domain_imgs[domain_id]
			self.label_list = self.train_domain_labels[domain_id]

			self.neg_imgs = self.img_list[:len(self.img_list)]
			self.neg_labels = self.label_list[:len(self.img_list)]

			seed = np.random.randint(1000)
			# seed = 777
			np.random.seed(seed)
			np.random.shuffle(self.neg_imgs)
			np.random.seed(seed)
			np.random.shuffle(self.neg_labels)
			# pdb.set_trace()

		elif phase == 'val':
			self.transform = transform
			self.img_list = self.val_img_list
			self.label_list = self.val_label_list

		elif phase == 'test':
			self.transform = transform
			self.img_list = self.test_img_list
			self.label_list = self.test_label_list

		# pdb.set_trace()

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		# if self.phase=='train':
		# image = image.resize((32, 32))
		# print(np.array(image).shape)
		label = self.label_list[item]
		if self.phase == 'train':
			neg_image = Image.open(self.neg_imgs[item]).convert('RGB')  # (C, H, W)
			# neg_image = neg_image.resize((32, 32))
			# print(np.array(neg_image).shape)
			neg_label = self.neg_labels[item]
			if self.transform is not None:
				neg_image = self.neg_transform(neg_image)
		else:
			neg_label = torch.Tensor(0)
			neg_image = torch.Tensor(0)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		transform_test = transforms.Compose([
	        transforms.ToTensor(),
	        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	    ])
		if self.transform is not None:
			# image = transform_test(image)
			image = self.transform(image)

		
		# return image and label
		# return image, self.label_list[item]
		return image, label, neg_image, neg_label

	def __len__(self):
		return len(self.img_list)


class rtMNIST(Dataset):
	def __init__(self, test_domain, train_domains, num_samples=10, transform=None):
		# assert phase in ['train', 'val', 'test']
		# self.domain_list = ['0', '15', '30', '45', '60', '75', '90']
		# for te_dom in test_domain:
		# 	self.domain_list.remove(te_dom)
		self.domain_list = train_domains
		self.num_domains = len(train_domains)
		self.num_samples = num_samples
		# assert 
		self.num_domains = len(self.domain_list)

		data_path = '../mnist/'

		self.sample_list = []

		self.infer_imgs = []
		self.infer_labels = []

		# self.num_imgs = []
		f = open('../mnist/' + self.domain_list[-1] + '_train_imgs.txt', 'r')
		lines = f.readlines()
		all_training_imgs = []
		all_training_labels = []
		
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			all_training_imgs.append(img[2:])
			all_training_labels.append(int(label))

		seed = 777
		np.random.seed(seed)
		np.random.shuffle(all_training_imgs)
		np.random.seed(seed)
		np.random.shuffle(all_training_labels)

		training_imgs = all_training_imgs[:10000]
		training_labels = all_training_labels[:10000]
		# pdb.set_trace()

		for i in range(len(self.domain_list)):
			samples = {}
			# pdb.set_trace()
			# domain_imgs = {}
			for j in range(10000):
				if training_labels[j] not in samples.keys():
					samples[training_labels[j]] = []
				samples[training_labels[j]].append(data_path + self.domain_list[i] + training_imgs[j][2:])
			self.sample_list.append(samples)

			# pdb.set_trace()

			for k in range(len(samples.keys())):
				self.infer_imgs = self.infer_imgs + samples[k][:num_samples]   # 20 samples for center feature during test
				self.infer_labels = self.infer_labels + [k] * num_samples

		# pdb.set_trace()

	def reset(self, phase, transform=None):
		# pdb.set_trace()
		self.phase = phase
		self.transform = transform
		if phase == 'train':
			self.img_list = []
			self.label_list = []
			for i in range(self.num_domains):
				for j in range(10):
					# pdb.set_trace()
					np.random.shuffle(self.sample_list[i][j])
					self.img_list = self.img_list + self.sample_list[i][j][:self.num_samples]
					self.label_list = self.label_list + [j] * self.num_samples

		else:
			self.img_list = self.infer_imgs
			self.label_list = self.infer_labels

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item])#.convert('L')  # (C, H, W)
		image = image.resize((28, 28))
		# image = np.array(image)
		# image = torch.FloatTensor(image).view(1, 28, 28)
		# image = torch.FloatTensor(image)
		# print(image)
		# print(np.array(image).shape)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# print(image.shape)
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)


class rtFMNIST(Dataset):
	def __init__(self, test_domain, train_domains, num_samples=10, transform=None):
		# assert phase in ['train', 'val', 'test']
		# self.domain_list = ['0', '15', '30', '45', '60', '75', '90']
		# for te_dom in test_domain:
		# 	self.domain_list.remove(te_dom)
		self.domain_list = train_domains
		self.num_domains = len(train_domains)
		self.num_samples = num_samples
		# assert 
		self.num_domains = len(self.domain_list)

		data_path = '../fashion_MNIST/'

		self.sample_list = []

		self.infer_imgs = []
		self.infer_labels = []

		# self.num_imgs = []
		f = open('../fashion_MNIST/' + self.domain_list[-1] + '_train_imgs.txt', 'r')
		lines = f.readlines()
		all_training_imgs = []
		all_training_labels = []
		
		for line in lines:
			[img, label] = line.strip('\n').split(' ')
			all_training_imgs.append(img[2:])
			all_training_labels.append(int(label))

		seed = 777
		np.random.seed(seed)
		np.random.shuffle(all_training_imgs)
		np.random.seed(seed)
		np.random.shuffle(all_training_labels)

		training_imgs = all_training_imgs[:10000]
		training_labels = all_training_labels[:10000]
		# pdb.set_trace()

		for i in range(len(self.domain_list)):
			samples = {}
			# pdb.set_trace()
			# domain_imgs = {}
			for j in range(10000):
				if training_labels[j] not in samples.keys():
					samples[training_labels[j]] = []
				samples[training_labels[j]].append(data_path + self.domain_list[i] + training_imgs[j][2:])
			self.sample_list.append(samples)

			# pdb.set_trace()

			for k in range(len(samples.keys())):
				self.infer_imgs = self.infer_imgs + samples[k][:num_samples]   # 20 samples for center feature during test
				self.infer_labels = self.infer_labels + [k] * num_samples

		# pdb.set_trace()

	def reset(self, phase, transform=None):
		# pdb.set_trace()
		self.phase = phase
		self.transform = transform
		if phase == 'train':
			self.img_list = []
			self.label_list = []
			for i in range(self.num_domains):
				for j in range(10):
					# pdb.set_trace()
					np.random.shuffle(self.sample_list[i][j])
					self.img_list = self.img_list + self.sample_list[i][j][:self.num_samples]
					self.label_list = self.label_list + [j] * self.num_samples

		else:
			self.img_list = self.infer_imgs
			self.label_list = self.infer_labels

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item])#.convert('L')  # (C, H, W)
		image = image.resize((28, 28))
		# image = np.array(image)
		# image = torch.FloatTensor(image).view(1, 28, 28)
		# image = torch.FloatTensor(image)
		# print(image)
		# print(np.array(image).shape)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# print(image.shape)
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)


class rtSVHN(Dataset):
	def __init__(self, train_domain, test_domain, num_samples=10, num_domains=1, transform=None):
		# assert phase in ['train', 'val', 'test']
		self.domain_list = ['all']
		self.num_domains = num_domains
		self.num_samples = num_samples
		# assert 
		self.num_domains = len(self.domain_list)

		self.sample_list = []

		self.infer_imgs = []
		self.infer_labels = []

		# self.num_imgs = []
		for i in range(len(self.domain_list)):
			f = open('../svhn/' + self.domain_list[i] + '.txt', 'r')
			lines = f.readlines()
			samples = {}
			# domain_imgs = {}
			for line in lines:
				[img, label] = line.strip('\n').split(' ')
				label = int(label)
				if label not in samples.keys():
					samples[label] = []
				samples[label].append('../svhn/' + img)
			self.sample_list.append(samples)

			# pdb.set_trace()

			for j in range(len(samples.keys())):
				self.infer_imgs = self.infer_imgs + samples[j][:num_samples]   # 20 samples for center feature during test
				self.infer_labels = self.infer_labels + [j] * num_samples
		# pdb.set_trace()


	def reset(self, phase, transform=None):
		# pdb.set_trace()
		self.phase = phase
		self.transform = transform
		if phase == 'train':
			self.img_list = []
			self.label_list = []
			for i in range(self.num_domains):
				for j in range(10):
					# pdb.set_trace()
					np.random.shuffle(self.sample_list[i][j])
					self.img_list = self.img_list + self.sample_list[i][j][:self.num_samples]
					self.label_list = self.label_list + [j] * self.num_samples

		else:
			self.img_list = self.infer_imgs
			self.label_list = self.infer_labels

		# pdb.set_trace()
		assert len(self.img_list)==len(self.label_list)

	def __getitem__(self, item):
		# image
		image = Image.open(self.img_list[item]).convert('RGB')  # (C, H, W)
		image = image.resize((32, 32))
		# image = np.array(image)
		# image = torch.FloatTensor(image).view(1, 28, 28)
		# image = torch.FloatTensor(image)
		# print(image)
		# print(np.array(image).shape)
		# image = image.resize((224, 224))
		# image = cv2.imread(self.img_list[item])[::-1]
		# pdb.set_trace()
		if self.transform is not None:
			image = self.transform(image)

		label = self.label_list[item]
		# print(image.shape)
		# return image and label
		# return image, self.label_list[item]
		return image, label

	def __len__(self):
		return len(self.img_list)