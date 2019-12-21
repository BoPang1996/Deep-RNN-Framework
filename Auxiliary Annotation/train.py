from torch import nn
from torch import optim
from torch.autograd import Variable
from data import load_data
from model import PolygonNet
import torch.utils.data
import numpy as np
import argparse
import os


def loss_overlap_coherence_function(pre, cur):
	mse_loss = nn.MSELoss()
	return mse_loss(cur, pre.detach())


parser = argparse.ArgumentParser(description='manual to train script')
parser.add_argument('--gpu_id', nargs='+', type=int)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--pretrained', type=str, default='False')
parser.add_argument('--num', type=int, default=50000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--opt_step', type=int, default=16)
args = parser.parse_args()

devices = args.gpu_id
batch_size = args.batch_size
num = args.num
lr = args.lr
opt_step = args.opt_step

lbd = 0.8
epoch_r = 5
loss_step = 100
root_path = "/Disk8/kevin/Cityscapes/"
user_path = "/home/kevin/polygon-release"

Dataloader = load_data(num, 'train', 60, root_path, batch_size)
len_dl = len(Dataloader)

torch.cuda.set_device(devices[0])
net = PolygonNet().cuda()
net = nn.DataParallel(net, device_ids=devices)

mapping_location = {'cuda:1': 'cuda:' + str(devices[0]), 'cuda:2': 'cuda:' + str(devices[0]),
                    'cuda:3': 'cuda:' + str(devices[0]), 'cuda:4': 'cuda:' + str(devices[0]),
                    'cuda:5': 'cuda:' + str(devices[0]), 'cuda:6': 'cuda:' + str(devices[0]),
                    'cuda:7': 'cuda:' + str(devices[0]), 'cuda:0': 'cuda:' + str(devices[0])}

if args.pretrained == 'True':
	print('Loading model...')
	net.load_state_dict(torch.load(root_path + 'save/4_50000.pth', map_location=mapping_location))
else:
	os.system('rm -rf {}/log'.format(user_path))

if not os.path.isfile('{}/control'.format(user_path)):
	os.system('touch {}/control'.format(user_path))

print('finished')

net.train()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.00001)  # update
optimizer.zero_grad()

dtype = torch.cuda.FloatTensor
dtype_t = torch.cuda.LongTensor

for epoch in range(epoch_r):
	accur = []

	for step, data in enumerate(Dataloader):
		x = Variable(data[0].type(dtype)).cuda()

		x1 = torch.unsqueeze(data[1], dim=1).repeat(1, data[2].shape[1], 1)
		x1 = torch.squeeze(x1, dim=0)
		x1 = Variable(x1.type(dtype)).cuda()

		x2 = torch.squeeze(data[2], dim=0)
		x2 = Variable(x2.type(dtype)).cuda()

		x3 = torch.squeeze(data[3], dim=0)
		x3 = Variable(x3.type(dtype)).cuda()

		ta = torch.squeeze(data[4], dim=0)
		ta = Variable(ta.type(dtype_t)).cuda()

		overlap = torch.squeeze(data[5], dim=0)
		overlap = Variable(overlap.type(dtype)).cuda()

		r = net(x, x1, x2, x3)

		result = r.contiguous().view(-1, 28 * 28 + 3)
		target = ta.contiguous().view(-1)
		_result = r.contiguous().view(data[2].shape[1], -1, 28 * 28 + 3)

		loss_coherence = torch.zeros(1).cuda()
		for idx in range(_result.size()[0] - 1):
			if overlap[idx].int() != 0:
				loss_coherence += loss_overlap_coherence_function(
					nn.Softmax(dim=1)(_result[idx, - overlap[idx].int():]),
					nn.Softmax(dim=1)(_result[idx + 1, :overlap[idx].int()]))
		loss_coherence /= result.size()[0]

		loss_objective = loss_function(result, target)

		loss = lbd * loss_coherence + loss_objective

		loss.backward()

		result_index = torch.from_numpy(np.argmax(result.data.cpu().numpy(), axis=1)).cuda()
		correct = (target == result_index).type(dtype).sum().item()
		acc = correct * 1.0 / target.shape[0]

		print("result_index: {}".format(result_index))
		print("target: {}".format(target))

		accur.append(acc)

		if step % (opt_step - 1) == 0:
			optimizer.step()
			optimizer.zero_grad()

		if step % loss_step == 0:
			mAccuarcy = np.mean(np.array(accur))
			accur = []

			with open('{}/log'.format(user_path), 'a') as outfile:
				print('Epoch: {}  Step: {}  Loss_C: {}  Loss_O: {}  Loss: {} mAccuracy: {}'.format(epoch,
				                                                                                   step,
				                                                                                   loss_coherence.data.cpu().numpy()[
					                                                                                   0],
				                                                                                   loss_objective.data.cpu().numpy(),
				                                                                                   loss.data.cpu().numpy()[
					                                                                                   0],
				                                                                                   mAccuarcy),
				      file=outfile)

			with open('{}/control'.format(user_path), "r") as infile:
				content = infile.readline()
				if content == "stop" or content == "stop\n":
					print("Saving Model...")
					torch.save(net.state_dict(), root_path + 'save/model' + '_' + str(num) + '.pth')
					print("Saving Finished!")
					exit()
				elif content == "save" or content == "save\n":
					print("Saving Model...")
					torch.save(net.state_dict(), root_path + 'save/model' + '_' + str(num) + '.pth')
					print("Saving Finished!")

	if len_dl > 200:
		torch.save(net.state_dict(), root_path + 'save/' + str(epoch) + '_' + str(num) + '.pth')
