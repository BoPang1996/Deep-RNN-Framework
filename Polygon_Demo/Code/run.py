# -*- coding: UTF-8 -*-
# 作者：曹瀚文

import torch.utils.data
import argparse
from model import PolygonNet
import os
from PIL import Image
from torchvision import transforms
from glob import glob


def deep_rnn_annotate():
	parser = argparse.ArgumentParser(description='manual to test script')
	# parser.add_argument('--gpu_id', nargs='+', type=int)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--model', type=str, default='./Polygon_deep_RNN.pth')
	args = parser.parse_args()

	devices = [0]
	batch_size = args.batch_size

	root_path = "../images"
	img_file = os.path.join(root_path, 'save.jpg')
	# mapping_location = {'cuda:0': 'cuda:' + str(devices[0])}
	img = Image.open(img_file).convert('RGB')
	img_len = img.size[0]
	img_width = img.size[1]
	transform = transforms.Compose([
		transforms.Resize((224,224)), # 只能对PIL图片进行裁剪
		transforms.ToTensor(),
		]
	)
	img = transform(img)
	img = img.unsqueeze(0)
	# torch.cuda.set_device(devices[0])
	net = PolygonNet(load_vgg=False)
	# net = nn.DataParallel(net, device_ids=devices).cuda()
	net.load_gpu_version(torch.load(args.model, map_location='cpu'))
	# print('finished')

	net.eval()

	dtype = torch.FloatTensor
	dtype_t = torch.LongTensor

	# output_path = "D:\\MyProject\\annotation_helper\\output\\output.txt"
	# os.system("rm -rf {}".format(output_path))

	# selected_classes = ['bicycle', 'bus', 'person', 'train', 'truck', 'motorcycle', 'car', 'rider']


	total = []
	total_wto_error = []
	error_num = 0
	error_flag = False

	with torch.no_grad():
		result = net.test(img.type(dtype), 60)
		labels_p = result.cpu().numpy()
		# print(labels_p)

		for i in range(result.shape[0]):
			vertices1 = []

			k = 0
			for k in range(len(labels_p[i])):
				if labels_p[i][k] == 784:
					k += 1
				else:
					break

			for p in range(k, len(labels_p[i])):
				if labels_p[i][p] == 784:
					break
				vertex = (round(((labels_p[i][p] % 28) * 8.0 + 4) * img_len / 224),
						  float(round((max(0, (int(labels_p[i][p] / 28) - 1)) * 8.0 + 4) * img_width / 224)))
				vertices1.append(vertex)

			# with open(output_path, 'a') as outfile:
			# 	print(vertices1, file=outfile)
			# print(type(vertices1))
			ret = []
			for i in range(len(vertices1)):
				ret.append(vertices1[i][0])
				ret.append(vertices1[i][1])
			# print(ret)
			return ret


if __name__ == "__main__":
	deep_rnn_annotate()
