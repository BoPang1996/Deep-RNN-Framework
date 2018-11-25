# -*- coding: UTF-8 -*-
import torch
from torch.autograd import Variable
from network import *

import UCF101_train
from utils import *
from torch import nn
from tqdm import tqdm
import os
from sync_batchnorm import DataParallelWithCallback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--LR', type=list, default=[1e-4, 1e-4, 1e-4], help='learning rate')  # start from 1e-4  5e-6
parser.add_argument('--EPOCH', type=int, default=30, help='epoch')
parser.add_argument('--slice_num', type=int, default=7, help='how many slices to cut')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--frame_num', type=int, default=10, help='how many frames in a slice')
parser.add_argument('--anticipation', action='store_true', help='whether commit the anticipation task')
# # 1000
# parser.add_argument('--model_path', type=str, default='/Disk1/poli/models/ShortRNN/UCF101', help='model_path')
# parser.add_argument('--model_name', type=str, default='recognition/encoding/checkpint_control_encoding', help='model name')
# parser.add_argument('--video_path', type=str, default='/Disk1/UCF101', help='video path')
# 400
parser.add_argument('--model_path', type=str, default='/Disk8/poli/models/ShortRNN/UCF101/recognition_RBM', help='model_path')
parser.add_argument('--model_name', type=str, default='checkpoint', help='model name')
parser.add_argument('--video_path', type=str, default='/Disk1/UCF101', help='video path')

parser.add_argument('--class_num', type=int, default=101, help='class num')
parser.add_argument('--device_id', type=list, default=[0,1,2,3], help='GPU device ID')
parser.add_argument('--resume', action='store_true', help='whether resume')
parser.add_argument('--dropout', type=list, default=[0.2, 0.2, 0.5], help='dropout')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--saveInter', type=int, default=1, help='how many epoch to save once')
parser.add_argument('--TemporalDropout', type=float, default=1.0, help='propabaility of Temporal Dropout')
parser.add_argument('--img_size', type=int, default=224, help='image size')
parser.add_argument('--logName', type=str, default='logs_res18', help='log dir name')
parser.add_argument('--overlap_rate', type=float, default=0.25, help='the overlap rate of the overlap coherence training scheme')
parser.add_argument('--lambdaa', type=float, default=0.0, help='weight of the overlap coherence loss')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.device_id[0])


# ######################## Build Model ########################################
print('Building model')
model = actionModel(opt.class_num, batch_norm=True, dropout=opt.dropout, detachout=opt.TemporalDropout)
model = DataParallelWithCallback(model, device_ids=opt.device_id).cuda()
print("Channels: " + str(model.module.channels))
optimizer = torch.optim.Adam([{'params':model.module.vgg.parameters(), 'lr': opt.LR[0]},
                              {'params': model.module.RNN.parameters(), 'lr': opt.LR[1]},
                              {'params': model.module.ShortCut.parameters(), 'lr': opt.LR[1]},
                              {'params':model.module.classifier.parameters(), 'lr': opt.LR[2]}
                              ], lr=opt.LR[-1], weight_decay=opt.weight_decay)
optimizer.zero_grad()
loss_classification_func = nn.CrossEntropyLoss(reduce=True)

def loss_overlap_coherence_func(pre, cur):
    loss = nn.MSELoss()
    return loss(cur, pre.detach())

def anticipation_loss_func(out_beforeMerge, b_action):
    loss_for_anticipation = Variable(
        torch.from_numpy(np.zeros((out_beforeMerge.shape[0], out_beforeMerge.shape[1])))).cuda().float()
    for i in range(out_beforeMerge.shape[0]):
        for j in range(out_beforeMerge.shape[1]):
            if (j != 0):
                if (b_action[i][j - 1] == b_action[i][j]):
                    loss_for_anticipation[i, j] = max(0, (
                            max(out_beforeMerge[i, 0:j, b_action[i][j].long()]) - out_beforeMerge[
                        i, j, b_action[i][j].long()]))
                else:
                    loss_for_anticipation[i, j] = out_beforeMerge[i, j, b_action[i][j - 1].long()]

    return 2 * torch.mean(loss_for_anticipation)

# ####################### Resume ################################################
resume_epoch = 0
if opt.resume:
    print("loading model")
    checkpoint = torch.load(opt.model_path + '/' + opt.model_name, map_location={'cuda:1': 'cuda:' + str(opt.device_id[0]),
                                                                         'cuda:2': 'cuda:' + str(opt.device_id[0]),
                                                                         'cuda:3': 'cuda:' + str(opt.device_id[0]),
                                                                         'cuda:4': 'cuda:' + str(opt.device_id[0]),
                                                                         'cuda:5': 'cuda:' + str(opt.device_id[0]),
                                                                         'cuda:6': 'cuda:' + str(opt.device_id[0]),
                                                                         'cuda:7': 'cuda:' + str(opt.device_id[0]),
                                                                         'cuda:8': 'cuda:' + str(opt.device_id[0])})
    model_state_dict = model.state_dict()
    for para in model_state_dict:
        model_state_dict[para] = checkpoint['model'][para]
    model.load_state_dict(model_state_dict)
    try:
        optimizer.load_state_dict(checkpoint['opt'])
    except:
        pass
    for group_id, param_group in enumerate(optimizer.param_groups):
        if group_id == 0:
            param_group['lr'] = opt.LR[0]*0.5
        elif group_id == 1:
            param_group['lr'] = opt.LR[1]*0.5
        elif group_id == 2:
            param_group['lr'] = opt.LR[2]*0.5
    resume_epoch = checkpoint['epoch']
    print('Finish Loading')
    del checkpoint


# ############################### Train ###########################################
print("START")
model.train()
predict_for_mAP = []
label_for_mAP = []


UCF101Loader = torch.utils.data.DataLoader(
    UCF101_train.UCF101(video_path=opt.video_path, frame_num=opt.frame_num, batch_size=opt.batch_size, img_size=224, slice_num=opt.slice_num, overlap_rate=opt.overlap_rate),
    batch_size=1, shuffle=True, num_workers=16)

print('Total Steps:' + str(len(UCF101Loader)))

for epoch in range(resume_epoch, opt.EPOCH):
    predict_for_mAP = []
    label_for_mAP = []

    for step, (x, _, overlap_frame_num, action) in enumerate(UCF101Loader):  # gives batch data
        x = x[0]
        action = action[0]
        overlap_frame_num = overlap_frame_num[0]

        for slice in range(x.shape[0]):
            b_x = Variable(x[slice]).cuda()
            b_action = Variable(action[slice]).cuda()

            out, out_beforeMerge = model(b_x.float())  # rnn output

            for batch in range(len(out)):
                predict_for_mAP.append(out[batch].data.cpu().numpy())
                label_for_mAP.append(b_action[batch][-1].data.cpu().numpy())

            # ###################### overlap coherence loss #######################################################################################
            loss_coherence = torch.zeros(1).cuda()

            # claculate the coherence loss with the previous clip and current clip
            if slice != 0:
                for b in range(out.size()[0]):
                    loss_coherence += loss_overlap_coherence_func(old_overlap[b],
                                                             nn.Softmax(dim=1)(out_beforeMerge[b, : overlap_frame_num[slice, b, 0].int()]))
                loss_coherence = loss_coherence / out.size()[0]

            # record the previous clips output
            old_overlap = []
            for b in range(out.size()[0]):
                old_overlap.append(nn.Softmax(dim=1)(out_beforeMerge[b, -overlap_frame_num[slice, b, 0].int():]))
            #######################################################################################################################################

            # anticipation loss
            loss_anticipation = torch.zeros(1).cuda()
            if opt.anticipation:
                loss_anticipation = anticipation_loss_func(out_beforeMerge, b_action)

            # classification loss
            loss_classification = loss_classification_func(out, b_action[:, -1].long()) + opt.lambdaa * loss_coherence

            loss = loss_classification + loss_coherence + loss_anticipation
            loss.backward()

        predict_for_mAP = predict_for_mAP
        label_for_mAP = label_for_mAP
        mAPs = mAP(predict_for_mAP, label_for_mAP,'sm')
        acc = accuracy(predict_for_mAP, label_for_mAP, 'sm')
        print("Epoch: " + str(epoch) + " step: " + str(step) + " Loss: " +str(loss.data.cpu().numpy()) + " mAP: " + str(mAPs) + " acc: " + str(acc))

        for p in model.parameters():
            p.grad.data.clamp_(min=-5, max=5)

        if step % 4 == 3:
            optimizer.step()
            optimizer.zero_grad()

        predict_for_mAP = []
        label_for_mAP = []


    if epoch % opt.saveInter == 0:
        print('Saving')
        torch.save({'model':model.state_dict(), 'epoch': epoch, 'opt':optimizer.state_dict()}, opt.model_path + '/' + opt.model_name + '_' + str(epoch))


