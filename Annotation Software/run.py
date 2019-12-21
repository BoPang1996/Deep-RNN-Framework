import torch.utils.data
import argparse
from model import PolygonNet
import os
from PIL import Image
from torchvision import transforms
from glob import glob


def deep_rnn_annotate():
	parser = argparse.ArgumentParser(description='manual to test script')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--model', type=str, default='./Polygon_deep_RNN.pth')
	args = parser.parse_args()

	devices = [0]
	batch_size = args.batch_size

	root_path = "../images"
	img_file = os.path.join(root_path, 'save.jpg')
	img = Image.open(img_file).convert('RGB')
	img_len = img.size[0]
	img_width = img.size[1]
	transform = transforms.Compose([
		transforms.Resize((224,224)), 
		transforms.ToTensor(),
		]
	)
	img = transform(img)
	img = img.unsqueeze(0)
	net = PolygonNet(load_vgg=False)
	net.load_gpu_version(torch.load(args.model, map_location='cpu'))

	net.eval()

	dtype = torch.FloatTensor
	dtype_t = torch.LongTensor

	total = []
	total_wto_error = []
	error_num = 0
	error_flag = False

	with torch.no_grad():
		result = net.test(img.type(dtype), 60)
		labels_p = result.cpu().numpy()

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

			ret = []
			for i in range(len(vertices1)):
				ret.append(vertices1[i][0])
				ret.append(vertices1[i][1])
			
			return ret


if __name__ == "__main__":
	deep_rnn_annotate()
