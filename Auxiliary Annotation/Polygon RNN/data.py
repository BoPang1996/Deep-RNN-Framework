import json
import torch
import numpy as np
from torch import utils
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image


class test_dataset(Dataset):
	def __init__(self, data_num, root_path, transform=None):
		self.num = data_num
		self.root_path = root_path
		self.transform = transform

	def __getitem__(self, index):
		img_file = self.root_path + 'new_img/val/{}.png'.format(index)

		img = Image.open(img_file).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)

		return (index, img)

	def __len__(self):
		return self.num


class newdataset(Dataset):
	def __init__(self, data_num, data_set, len_s, root_path, transform=None):
		self.num = data_num
		self.dataset = data_set
		self.length = len_s
		self.root_path = root_path
		self.transform = transform

	def __getitem__(self, index):
		img_name = self.root_path + 'new_img/{}/{}.png'.format(self.dataset, index)
		label_name = self.root_path + 'new_label/{}/{}.json'.format(self.dataset, index)
		try:
			img = Image.open(img_name).convert('RGB')
		except FileNotFoundError:
			return None
		assert not (img is None)

		json_file = json.load(open(label_name))
		point_num = len(json_file['polygon'])
		polygon = np.array(json_file['polygon'])
		point_count = 2

		label_array = np.zeros([self.length, 28 * 28 + 3])
		label_index_array = np.zeros([self.length])

		if point_num < self.length - 3:
			for points in polygon:
				index_a = int(points[0] / 8)
				index_b = int(points[1] / 8)
				index = index_b * 28 + index_a
				label_array[point_count, index] = 1
				label_index_array[point_count] = index
				point_count += 1
			label_array[point_count, 28 * 28] = 1
			label_index_array[point_count] = 28 * 28
			for kkk in range(point_count + 1, self.length):
				if kkk % (point_num + 3) == point_num + 2:
					index = 28 * 28
				elif kkk % (point_num + 3) == 0:
					index = 28 * 28 + 1
				elif kkk % (point_num + 3) == 1:
					index = 28 * 28 + 2
				else:
					index_a = int(polygon[kkk % (point_num + 3) - 2][0] / 8)
					index_b = int(polygon[kkk % (point_num + 3) - 2][1] / 8)
					index = index_b * 28 + index_a
				label_array[kkk, index] = 1
				label_index_array[kkk] = index
		else:
			scale = point_num * 1.0 / (self.length - 3)
			index_list = (np.arange(0, self.length - 3) * scale).astype(int)
			for points in polygon[index_list]:
				index_a = int(points[0] / 8)
				index_b = int(points[1] / 8)
				index = index_b * 28 + index_a
				label_array[point_count, index] = 1
				label_index_array[point_count] = index
				point_count += 1
			for kkk in range(point_count, self.length):
				index = 28 * 28
				label_array[kkk, index] = 1
				label_index_array[kkk] = index

		if self.transform is not None:
			img = self.transform(img)

		num_slice = 8

		return (img, label_array[2], data_divider(label_array[:-2], num_slice)[0],
		        data_divider(label_array[1:-1], num_slice)[0], data_divider(label_index_array[2:], num_slice)[0],
		        data_divider(label_array[:-2], num_slice)[1])

	def __len__(self):
		return self.num


def data_divider(label_array, num_slice):
	overlap = [0] * (num_slice - 1)
	start_point = [0] * num_slice
	end_point = [0] * num_slice

	length = label_array.shape[0]
	slice_length = int((length / num_slice) * 1.5)
	if (len(label_array.shape)) == 2:
		new_label_array = np.zeros([num_slice, slice_length, label_array.shape[1]])
	else:
		new_label_array = np.zeros([num_slice, slice_length])

	first_id = 0
	last_id = length - slice_length
	new_label_array[0] = label_array[:slice_length]
	new_label_array[-1] = label_array[last_id:]
	start_point[0] = 0
	end_point[0] = slice_length
	start_point[-1] = last_id
	end_point[-1] = length

	start_id = 0
	stride = (last_id - first_id) / (num_slice - 1)
	for i in range(1, num_slice - 1):
		new_label_array[i] = label_array[int(start_id + i * stride): int(start_id + i * stride + slice_length)]
		start_point[i] = start_id + i * stride
		end_point[i] = start_id + i * stride + slice_length

	for i in range(num_slice - 1):
		if end_point[i] > start_point[i + 1]:
			overlap[i] = end_point[i] - start_point[i + 1]
		else:
			overlap[i] = 0

	return new_label_array, torch.FloatTensor(overlap)


def load_test_data(data_num, root_path, batch_size=2):
	trans = transforms.Compose([transforms.ToTensor(),
	                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	datas = test_dataset(data_num, root_path, trans)
	Dataloader = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=False, drop_last=False,
	                                         num_workers=8)
	return Dataloader


def load_data(data_num, data_set, len_s, root_path, batch_size=1):
	trans = transforms.Compose([transforms.ToTensor(),
	                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	datas = newdataset(data_num, data_set, len_s, root_path, trans)
	Dataloader = torch.utils.data.DataLoader(datas, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
	return Dataloader
