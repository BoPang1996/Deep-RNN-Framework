# -*- coding: UTF-8 -*-
import torch
from torch.autograd import Variable
from network import *
import UCF101_test
from utils import *
from tqdm import tqdm
import argparse
from sync_batchnorm import DataParallelWithCallback

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--model_path', type=str, default='/Disk8/poli/models/ShortRNN/UCF101/recognition_RBM', help='model_path')
parser.add_argument('--model_name', type=str, default='checkpoint', help='model name')
parser.add_argument('--video_path', type=str, default='/Disk1/UCF101', help='video path')
parser.add_argument('--class_num', type=int, default=101, help='class num')
parser.add_argument('--device_id', type=list, default=[0,1], help='learning rate')
parser.add_argument('--frame_num', type=int, default=30, help='how many frames used in test')
parser.add_argument('--anticipation', action='store_true', help='whether commit the anticipation task')

args = parser.parse_args()


torch.cuda.set_device(args.device_id[0])
if args.anticipation:
    args.frame_num = 2


print('Building model')
model = actionModel(args.class_num, batch_norm=True, dropout=[0, 0, 0])
model = DataParallelWithCallback(model, device_ids=args.device_id).cuda()


print("loading model")
checkpoint = torch.load(args.model_path + '/' + args.model_name, map_location={'cuda:1': 'cuda:' + str(args.device_id[0]),
                                                                     'cuda:2': 'cuda:' + str(args.device_id[0]),
                                                                     'cuda:3': 'cuda:' + str(args.device_id[0]),
                                                                     'cuda:4': 'cuda:' + str(args.device_id[0]),
                                                                     'cuda:5': 'cuda:' + str(args.device_id[0]),
                                                                     'cuda:6': 'cuda:' + str(args.device_id[0]),
                                                                     'cuda:7': 'cuda:' + str(args.device_id[0]),
                                                                     'cuda:0': 'cuda:' + str(args.device_id[0])})
pre_train = checkpoint['model']
model_dict = model.state_dict()
for para in pre_train:
    if para in model_dict:
        model_dict[para] = pre_train[para]
model.load_state_dict(model_dict)
print('Finish Loading')
del checkpoint, pre_train, model_dict
print("Model: " + str(args.model_name))


predict_for_mAP = []
label_for_mAP = []

print("START")

UCF101Loader_test = torch.utils.data.DataLoader(
    UCF101_test.UCF101(video_path=args.video_path, frame_num=args.frame_num, img_size=224, anticipation=args.anticipation),
    batch_size=args.batch_size, shuffle=True, num_workers=0)

print(len(UCF101Loader_test))


with torch.no_grad():
    model.eval()
    predict_for_mAP = []
    label_for_mAP = []
    print("TESTING")

    for step, (x, _, _, action) in tqdm(enumerate(UCF101Loader_test)):  # gives batch data
        b_x = Variable(x).cuda()
        b_action = Variable(action).cuda()

        out, _ = model(b_x.float())  # rnn output

        for batch in range(len(out)):
            predict_for_mAP.append(out[batch].data.cpu().numpy())
            label_for_mAP.append(b_action[batch][-1].data.cpu().numpy())

        if step % 50 == 0:
            MAP = mAP(np.array(predict_for_mAP), np.array(label_for_mAP), 'Lsm')
            acc = accuracy(np.array(predict_for_mAP), np.array(label_for_mAP), 'Lsm')
            print(" mAP: " + str(MAP)[0:7] + '  ' + 'accuracy: ' + str(acc)[0:7])

    predict_for_mAP = np.array(predict_for_mAP)
    label_for_mAP = np.array(label_for_mAP)

    MAP = mAP(predict_for_mAP, label_for_mAP, 'Lsm')
    acc = accuracy(predict_for_mAP, label_for_mAP, 'Lsm')

    print("mAP: " + str(MAP) + '  ' + 'accuracy: ' + str(acc))


