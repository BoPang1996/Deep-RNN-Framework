from torch import nn
import torch.utils.data
import argparse
from data import load_test_data
from model import PolygonNet
from PIL import Image, ImageDraw
import numpy as np
import json
import os
from glob import glob

parser = argparse.ArgumentParser(description='manual to test script')
parser.add_argument('--gpu_id', nargs='+', type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model', type=str)
args = parser.parse_args()

devices = args.gpu_id
batch_size = args.batch_size

root_path = "/Disk8/kevin/Cityscapes/"
mapping_location = {'cuda:1': 'cuda:' + str(devices[0]), 'cuda:2': 'cuda:' + str(devices[0]),
                    'cuda:3': 'cuda:' + str(devices[0]), 'cuda:4': 'cuda:' + str(devices[0]),
                    'cuda:5': 'cuda:' + str(devices[0]), 'cuda:6': 'cuda:' + str(devices[0]),
                    'cuda:7': 'cuda:' + str(devices[0]), 'cuda:0': 'cuda:' + str(devices[0])}

torch.cuda.set_device(devices[0])
net = PolygonNet(load_vgg=False)
net = nn.DataParallel(net, device_ids=devices).cuda()
net.load_state_dict(torch.load(args.model, map_location=mapping_location))
print('finished')

net.eval()

dtype = torch.cuda.FloatTensor
dtype_t = torch.cuda.LongTensor

output_path = "/home/kevin/polygon-release/test_log"
os.system("rm -rf {}".format(output_path))

selected_classes = ['bicycle', 'bus', 'person', 'train', 'truck', 'motorcycle', 'car', 'rider']

iou = {}
iou_wto_error = {}
for cls in selected_classes:
	iou[cls] = []
	iou_wto_error[cls] = []

total = []
total_wto_error = []
error_num = 0
error_flag = False

with torch.no_grad():
	files = glob(root_path + 'new_img/val/*.png')
	Dataloader = load_test_data(len(files), root_path, batch_size)
	for step, data in enumerate(Dataloader):
		index, img = data

		result = net.module.test(img.type(dtype), 60)
		labels_p = result.cpu().numpy()

		for i in range(result.shape[0]):
			json_file = root_path + 'new_label/val/{}.json'.format(index[i])

			json_object = json.load(open(json_file))
			h = json_object['imgHeight']
			w = json_object['imgWidth']
			polygon = json_object['polygon']
			min_row = json_object['min_row']
			min_col = json_object['min_col']
			scale_w = json_object['scale_w']
			scale_h = json_object['scale_h']
			label_name = json_object['label']

			vertices1 = []
			vertices2 = []

			k = 0
			for k in range(len(labels_p[i])):
				if labels_p[i][k] == 784:
					k += 1
				else:
					break

			for p in range(k, len(labels_p[i])):
				if (labels_p[i][p] == 784):
					break
				vertex = (((labels_p[i][p] % 28) * 8.0 + 4) / scale_w + min_col,
				          ((int(labels_p[i][p] / 28)) * 8.0 + 4) / scale_h + min_row)
				vertices1.append(vertex)

			for points in polygon:
				vertex = (points[0], points[1])
				vertices2.append(vertex)

			try:
				img1 = Image.new('L', (w, h), 0)
				ImageDraw.Draw(img1).polygon(vertices1, outline=1, fill=1)
				mask1 = np.array(img1)
				img2 = Image.new('L', (w, h), 0)
				ImageDraw.Draw(img2).polygon(vertices2, outline=1, fill=1)
				mask2 = np.array(img2)

				nu = 0
				de = 0

				intersection = np.logical_and(mask1, mask2)
				union = np.logical_or(mask1, mask2)
				nu += np.sum(intersection)
				de += np.sum(union)
				IOU_ultimate = nu * 1.0 / de
			except:
				print("error")
				error_num += 1
				IOU_ultimate = 0
				error_flag = True

			iou[label_name].append(IOU_ultimate)
			total.append(IOU_ultimate)

			if not error_flag:
				iou_wto_error[label_name].append(IOU_ultimate)
				total_wto_error.append(IOU_ultimate)
			else:
				error_flag = False

			with open(output_path, 'a') as outfile:
				print("[{}] IoU: {}".format(index[i], IOU_ultimate), file=outfile)

			print("[{}] IoU: {}".format(index[i], IOU_ultimate))

	for cls in selected_classes:
		iou_mean = np.mean(np.array(iou[cls]))
		with open(output_path, 'a') as outfile:
			print("IoU of Class [" + cls + "]: " + str(iou_mean) + "		NUM = " + str(len(iou[cls])), file=outfile)
		print("IoU of Class [" + cls + "]: " + str(iou_mean) + "		NUM = " + str(len(iou[cls])))

	with open(output_path, 'a') as outfile:
		print("IoU of [Mean]: " + str(np.mean(np.array(total))) + "		TOTAL_NUM = " + str(len(total)),
		      file=outfile)
		print("IoU of [Mean](wto errors): " + str(
			np.mean(np.array(total_wto_error))) + "		TOTAL_NUM = " + str(len(total_wto_error)), file=outfile)
		print("error_num: " + str(error_num), file=outfile)
	print("IoU of [Mean]: " + str(np.mean(np.array(total))) + "		TOTAL_NUM = " + str(len(total)))
	print("IoU of [Mean](wto errors): " + str(np.mean(np.array(total_wto_error))) + "		TOTAL_NUM = " + str(
		len(total_wto_error)))
	print("error_num: {}".format(error_num))
