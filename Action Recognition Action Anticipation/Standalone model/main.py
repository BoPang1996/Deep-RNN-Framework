# -*- coding: UTF-8 -*-
import torch
from torch.autograd import Variable
from network import *
import Kinetics_test_dataset
import Kinetic_train_dataset
from utils import *
from torch import nn
from tqdm import tqdm
from sync_batchnorm import DataParallelWithCallback
import argparse
from tensorboardX import SummaryWriter


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--LR', type=list, default=[1e-4, 1e-4], help='learning rate')  # start from 1e-4
parser.add_argument('--EPOCH', type=int, default=30, help='epoch')
parser.add_argument('--slice_num', type=int, default=6, help='how many slices to cut')
parser.add_argument('--batch_size', type=int, default=40, help='batch_size')
parser.add_argument('--frame_num', type=int, default=5, help='how many frames in a slice')
parser.add_argument('--model_path', type=str, default='/Disk1/poli/models/DeepRNN/Kinetics_res18', help='model_path')
parser.add_argument('--model_name', type=str, default='checkpoint', help='model name')
parser.add_argument('--video_path', type=str, default='/home/poli/kinetics_scaled', help='video path')
parser.add_argument('--class_num', type=int, default=400, help='class num')
parser.add_argument('--device_id', type=list, default=[0,1,2,3], help='learning rate')
parser.add_argument('--resume', action='store_true', help='whether resume')
parser.add_argument('--dropout', type=list, default=[0.2, 0.5], help='dropout')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--saveInter', type=int, default=1, help='how many epoch to save once')
parser.add_argument('--TD_rate', type=float, default=0.0, help='propabaility of detachout')
parser.add_argument('--img_size', type=int, default=224, help='image size')
parser.add_argument('--syn_bn', action='store_true', help='use syn_bn')
parser.add_argument('--logName', type=str, default='logs_res18', help='log dir name')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--test', action='store_true', help='test the model')
parser.add_argument('--overlap_rate', type=float, default=0.25, help='the overlap rate of the overlap coherence training scheme')
parser.add_argument('--lambdaa', type=float, default=0.0, help='weight of the overlap coherence loss')

opt = parser.parse_args()
print(opt)

torch.cuda.set_device(opt.device_id[0])

# ######################## Module #################################
print('Building model')
model = actionModel(opt.class_num, batch_norm=True, dropout=opt.dropout, TD_rate=opt.TD_rate, image_size=opt.img_size, syn_bn=opt.syn_bn, test_scheme=3)
print(model)
if opt.syn_bn:
    model = DataParallelWithCallback(model, device_ids=opt.device_id).cuda()
else:
    model = torch.nn.DataParallel(model, device_ids=opt.device_id).cuda()
print("Channels: " + str(model.module.channels))

# ########################Optimizer#########################
optimizer = torch.optim.SGD([{'params': model.module.RNN.parameters(), 'lr': opt.LR[0]},
                             {'params': model.module.ShortCut.parameters(), 'lr': opt.LR[0]},
                             {'params': model.module.classifier.parameters(), 'lr': opt.LR[1]}
                             ], lr=opt.LR[1], weight_decay=opt.weight_decay, momentum=0.9)

# ###################### Loss Function ####################################
loss_classification_func = nn.NLLLoss(reduce=True)


def loss_overlap_coherence_func(pre, cur):
    loss = nn.MSELoss()
    return loss(cur, pre.detach())


# ###################### Resume ##########################################
resume_epoch = 0
resume_step = 0
max_test_acc = 0

if opt.resume or opt.test:
    print("loading model")
    checkpoint = torch.load(opt.model_path + '/' + opt.model_name, map_location={'cuda:0': 'cuda:' + str(opt.device_id[0]),
                                                                                 'cuda:1': 'cuda:' + str(opt.device_id[0]),
                                                                                 'cuda:2': 'cuda:' + str(opt.device_id[0]),
                                                                                 'cuda:3': 'cuda:' + str(opt.device_id[0]),
                                                                                 'cuda:4': 'cuda:' + str(opt.device_id[0]),
                                                                                 'cuda:5': 'cuda:' + str(opt.device_id[0]),
                                                                                 'cuda:6': 'cuda:' + str(opt.device_id[0]),
                                                                                 'cuda:7': 'cuda:' + str(opt.device_id[0])})

    model.load_state_dict(checkpoint['model'], strict=True)
    try:
        optimizer.load_state_dict(checkpoint['opt'], strict=True)
    except:
        pass
    for group_id, param_group in enumerate(optimizer.param_groups):
        if group_id == 0:
            param_group['lr'] = opt.LR[0]
        elif group_id == 1:
            param_group['lr'] = opt.LR[0]
        elif group_id == 2:
            param_group['lr'] = opt.LR[1]
    resume_epoch = checkpoint['epoch']
    if 'step' in checkpoint:
        resume_step = checkpoint['step'] + 1
    if 'max_acc' in checkpoint:
        max_test_acc = checkpoint['max_acc']
    print('Finish Loading')
    del checkpoint
# ###########################################################################

# training and testing
model.train()
predict_for_mAP = []
label_for_mAP = []

print("START")


KineticsLoader = torch.utils.data.DataLoader(
    Kinetic_train_dataset.Kinetics(video_path=opt.video_path + '/train_frames', frame_num=opt.frame_num, batch_size=opt.batch_size, img_size=opt.img_size, slice_num=opt.slice_num, overlap_rate=opt.overlap_rate),
    batch_size=1, shuffle=True, num_workers=8)

Loader_test = torch.utils.data.DataLoader(
    Kinetics_test_dataset.Kinetics(video_path=opt.video_path + '/val_frames', img_size=224, space=5,
                                   split_num=8, lenn=60, num_class=opt.class_num),
    batch_size=64, shuffle=True, num_workers=4)


tensorboard_writer = SummaryWriter(opt.logName, purge_step=resume_epoch * len(KineticsLoader) * opt.slice_num + (
                                              resume_step + resume_step) * opt.slice_num)


test = opt.test
for epoch in range(resume_epoch, opt.EPOCH):

    predict_for_mAP = []
    label_for_mAP = []

    for step, (x, _, overlap_frame_num, action) in enumerate(KineticsLoader):  # gives batch data

        if opt.train:
            if step + resume_step >= len(KineticsLoader):
                break
            x = x[0]
            action = action[0]
            overlap_frame_num = overlap_frame_num[0]

            c = [Variable(torch.from_numpy(np.zeros((x.shape[1], model.module.channels[layer + 1], model.module.input_size[layer], model.module.input_size[layer])))).cuda().float() for layer in range(model.module.RNN_layer)]
            for slice in range(x.shape[0]):
                b_x = Variable(x[slice]).cuda()
                b_action = Variable(action[slice]).cuda()

                out, out_beforeMerge, c = model(b_x.float(), c)  # rnn output
                for batch in range(len(out)):
                    predict_for_mAP.append(out[batch].data.cpu().numpy())
                    label_for_mAP.append(b_action[batch][-1].data.cpu().numpy())

                # ###################### overlap coherence loss #######################################################################################
                loss_coherence = torch.zeros(1).cuda()

                # claculate the coherence loss with the previous clip and current clip
                if slice != 0:
                    for b in range(out.size()[0]):
                        loss_coherence += loss_overlap_coherence_func(old_overlap[b],
                                                                      torch.exp(out_beforeMerge[b, : overlap_frame_num[slice, b, 0].int()]))
                    loss_coherence = loss_coherence / out.size()[0]

                # record the previous clips output
                old_overlap = []
                for b in range(out.size()[0]):
                    old_overlap.append(torch.exp(out_beforeMerge[b, -overlap_frame_num[slice, b, 0].int():]))
                #######################################################################################################################################

                loss_classification = loss_classification_func(out, b_action[:, -1].long())

                loss = loss_classification + opt.lambdaa * loss_coherence
                tensorboard_writer.add_scalar('train/loss', loss, epoch*len(KineticsLoader)*opt.slice_num + (step+resume_step)*opt.slice_num + slice)

                loss.backward(retain_graph=False)


            predict_for_mAP = predict_for_mAP
            label_for_mAP = label_for_mAP
            mAPs = mAP(predict_for_mAP, label_for_mAP, 'Lsm')
            acc = accuracy(predict_for_mAP, label_for_mAP, 'Lsm')
            tensorboard_writer.add_scalar('train/mAP', mAPs,
                                          epoch * len(KineticsLoader) * opt.slice_num + (
                                                  step + resume_step) * opt.slice_num + slice)
            tensorboard_writer.add_scalar('train/acc', acc,
                                          epoch * len(KineticsLoader) * opt.slice_num + (
                                                  step + resume_step) * opt.slice_num + slice)

            print("Epoch: " + str(epoch) + " step: " + str(step+resume_step) + " Loss: " + str(loss.data.cpu().numpy()) + " Loss_coherence: " + str(loss_coherence.data.cpu().numpy()) + " mAP: " + str(mAPs)[0:7] + " acc: " + str(acc)[0:7])

            for p in model.module.parameters():
                p.grad.data.clamp_(min=-5, max=5)

            if step % 2 == 1:
                optimizer.step()
                optimizer.zero_grad()

            predict_for_mAP = []
            label_for_mAP = []

        # ################################### test ###############################
        if (step + resume_step) % 700 == 699:
            test = True

        if test:
            print('Start Test')
            TEST_LOSS = AverageMeter()
            with torch.no_grad():
                model.eval()
                predict_for_mAP = []
                label_for_mAP = []
                print("TESTING")

                for step_test, (x, _, _, action) in tqdm(enumerate(Loader_test)):  # gives batch data
                    b_x = Variable(x).cuda()
                    b_action = Variable(action).cuda()

                    c = [Variable(torch.from_numpy(np.zeros((len(b_x), model.module.channels[layer + 1],
                                                             model.module.input_size[layer],
                                                             model.module.input_size[layer])))).cuda().float() for layer
                         in
                         range(model.module.RNN_layer)]
                    out, _, _ = model(b_x.float(), c)  # rnn output
                    loss = loss_classification_func(out, b_action[:, -1].long())
                    TEST_LOSS.update(val=loss.data.cpu().numpy())

                    for batch in range(len(out)):
                        predict_for_mAP.append(out[batch].data.cpu().numpy())
                        label_for_mAP.append(b_action[batch][-1].data.cpu().numpy())

                    if step_test % 50 == 0:
                        MAP = mAP(np.array(predict_for_mAP), np.array(label_for_mAP), 'Lsm')
                        acc = accuracy(np.array(predict_for_mAP), np.array(label_for_mAP), 'Lsm')
                        print(" Loss: " + str(TEST_LOSS.avg)[0:5] + '  ' + 'accuracy: ' + str(acc)[0:7])

                predict_for_mAP = np.array(predict_for_mAP)
                label_for_mAP = np.array(label_for_mAP)

                MAP = mAP(predict_for_mAP, label_for_mAP, 'Lsm')
                acc = accuracy(predict_for_mAP, label_for_mAP, 'Lsm')

                print("mAP: " + str(MAP) + '  ' + 'accuracy: ' + str(acc))

                if acc > max_test_acc:
                    print('Saving')
                    max_test_acc = acc
                    torch.save({'model': model.state_dict(), 'max_acc': max_test_acc, 'epoch': epoch, 'step': 0,
                                'opt': optimizer.state_dict()},
                               opt.model_path + '/' + opt.model_name + '_' + str(epoch) + '_' + str(max_test_acc)[0:6])
                model.train()

                test = False
                predict_for_mAP = []
                label_for_mAP = []

                if opt.test:
                    exit()

    if epoch % opt.saveInter == 0:
        print('Saving')
        torch.save({'model': model.state_dict(), 'max_acc': max_test_acc, 'epoch': epoch, 'step': 0, 'opt': optimizer.state_dict()}, opt.model_path + '/' + opt.model_name + '_' + str(epoch))

    resume_step = 0




