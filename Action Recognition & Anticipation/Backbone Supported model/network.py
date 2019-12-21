import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from sync_batchnorm import SynchronizedBatchNorm2d
import random
class SNN(nn.Module):

    def __init__(self, net_dict, batch_norm, num_action, dropout, RNN_layer, channels, test_scheme=1, detachout=0.0):

        super(SNN, self).__init__()

        self.dropout1 = nn.Dropout2d(p = dropout[0])
        self.dropout2 = nn.Dropout2d(p = dropout[1])
        self.detachout = detachout
        self.vgg = self._make_layer(net_dict[0], batch_norm).cuda()  # VGG
        self.init_weight(self.vgg)

        self.channel_times = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.RNN = nn.ModuleList([self.make_RNNCell(channels * (self.channel_times[0]), channels * (self.channel_times[1] + self.channel_times[0]), channels * self.channel_times[1], batch_norm, detachout), # comment for 3 layer
                                  self.make_RNNCell(channels * (self.channel_times[1]), channels * (self.channel_times[2] + self.channel_times[1]), channels * self.channel_times[2], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[2]), channels *  (self.channel_times[3] + self.channel_times[2]), channels * self.channel_times[3], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[3]), channels *  (self.channel_times[4] + self.channel_times[3]), channels * self.channel_times[4], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[4]), channels *  (self.channel_times[5] + self.channel_times[4]), channels * self.channel_times[5], batch_norm, detachout, pool=True),
                                  self.make_RNNCell(channels * (self.channel_times[5]), channels *  (self.channel_times[6] + self.channel_times[5]), channels * self.channel_times[6], batch_norm, detachout),  # comment for 3 layer
                                  self.make_RNNCell(channels * (self.channel_times[6]), channels *  (self.channel_times[7] + self.channel_times[6]), channels * self.channel_times[7], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[7]), channels *  (self.channel_times[8] + self.channel_times[7]), channels * self.channel_times[8], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[8]), channels *  (self.channel_times[9] + self.channel_times[8]), channels * self.channel_times[9], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[9]), channels *  (self.channel_times[10] + self.channel_times[9]), channels * self.channel_times[10], batch_norm, detachout, pool=True),
                                  self.make_RNNCell(channels * (self.channel_times[10]), channels *  (self.channel_times[11] + self.channel_times[10]), channels * self.channel_times[11], batch_norm, detachout),  # comment for 3 layer
                                  self.make_RNNCell(channels * (self.channel_times[11]), channels *  (self.channel_times[12] + self.channel_times[11]), channels * self.channel_times[12], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[12]), channels *  (self.channel_times[13] + self.channel_times[12]), channels* self.channel_times[13], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[13]), channels *  (self.channel_times[14] + self.channel_times[13]), channels* self.channel_times[14], batch_norm, detachout),
                                  self.make_RNNCell(channels * (self.channel_times[14]), channels *  (self.channel_times[15] + self.channel_times[14]), channels* self.channel_times[15], batch_norm, detachout, pool=True)])
        self.init_weight(self.RNN, xavier_gain=3.0)

        block = [[{'convsc_1': [channels, channels * self.channel_times[5], 1, 1, 0]}],
                 [{'convsc_2': [channels * self.channel_times[5], channels * self.channel_times[10], 1, 1, 0]}],
                 [{'convsc_3': [channels * self.channel_times[10], channels * self.channel_times[15], 1, 1, 0]}]]
        self.ShortCut = nn.ModuleList([self._make_layer(block[i], batch_norm) for i in range(len(block))])
        self.init_weight(self.ShortCut)

        self.classifier = nn.Sequential(nn.Linear(channels * self.channel_times[14], channels * 1), nn.ReLU(inplace=True), nn.Dropout(p=dropout[2]), nn.Linear(channels * 1, num_action))#, nn.LogSoftmax(dim=1))#nn.Softmax(dim=1))
        self.init_weight(self.classifier)


        self.num_action = num_action
        self.RNN_layer = RNN_layer
        self.channels = channels
        self.test_scheme = test_scheme

    def print_(self, strr):
        if self.training:
            print(strr)

    def init_weight(self, model, xavier_gain=1.0):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                #m.weight.data.normal_(0, 0.01)
                nn.init.xavier_normal_(m.weight.data, gain=xavier_gain)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, net_dict, batch_norm=False):

        layers = []
        length = len(net_dict)
        for i in range(length):
            one_layer = net_dict[i]
            key = one_layer.keys()[0]
            v = one_layer[key]

            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                if batch_norm:
                    layers += [conv2d, SynchronizedBatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


    def make_RNNCell(self, in_channel1, in_channel2, out_channel, batch_norm, detachout, pool=False):
        class RNN_cell(nn.Module):
            def __init__(self, in_channel, out_channel, batch_norm, pool):
                super(RNN_cell, self).__init__()
                self.outchannel = out_channel
                conv_R = nn.Conv2d(in_channels=in_channel1, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
                conv_T = nn.Conv2d(in_channels=in_channel2, out_channels=out_channel, kernel_size=3, stride=1,padding=1)
                layers_data = [conv_R,SynchronizedBatchNorm2d(out_channel)]#,nn.BatchNorm2d(out_channel * 4), nn.Dropout2d(p=dropout)]
                layers_ctrl = [conv_T,SynchronizedBatchNorm2d(out_channel)]

                self.conv_data = nn.Sequential(*layers_data)
                self.conv_ctrl = nn.Sequential(*layers_ctrl)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
                self.ispool = pool
                self.detachout = detachout


            def forward(self, x, c):
                input_data = x
                rand =random.random()
                if rand < self.detachout:
                    x_ctrl = x.detach()
                else:
                    x_ctrl = x
                input_ctrl = torch.cat((x_ctrl,c), 1)
                #input = torch.cat((x,c), 1)
                data = torch.tanh(self.conv_data(input_data))
                ctrl = torch.sigmoid(self.conv_ctrl(input_ctrl))

                output = data * ctrl

                return output, ctrl  #, insert_content

        return RNN_cell(in_channel2, out_channel, batch_norm, pool)



    def forward(self, x):
        out0 = Variable(torch.from_numpy(np.zeros((x.shape[0],x.shape[1], self.channels, x.shape[-1]/32, x.shape[-1]/32)))).cuda().float()
        for i in range(x.shape[1]):
            out0[:,i,:,:,:] = self.dropout1(self.vgg(x[:,i])) #batch, frame, channel, w, h

        output_ = out0
        output = []
        input_size =     [7, 7, 7, 7, 7, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2]
        out_size =       [7, 7, 7, 7, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 1]

        outForShortcut = [Variable(torch.from_numpy(np.zeros((x.shape[0], x.shape[1], self.channels* self.channel_times[0], out_size[0], out_size[0])))).cuda().float(),
                          Variable(torch.from_numpy(np.zeros((x.shape[0], x.shape[1], self.channels* self.channel_times[4], out_size[4], out_size[4])))).cuda().float(),
                          Variable(torch.from_numpy(np.zeros((x.shape[0], x.shape[1], self.channels* self.channel_times[9], out_size[9], out_size[9])))).cuda().float()]
        for layer in range(self.RNN_layer):
            c = Variable(torch.from_numpy(np.zeros((x.shape[0], self.channels * self.channel_times[layer],input_size[layer], input_size[layer])))).cuda().float()

            input = output_
            if layer == 0 or layer == 5 or layer == 10:
                outForShortcut[layer / 5] = input
            output_ = Variable(torch.from_numpy(np.zeros((x.shape[0], x.shape[1], self.channels * self.channel_times[layer], out_size[layer], out_size[layer])))).cuda().float()
            for frame in range(x.shape[1]):
                h, c = self.RNN[layer](input[:,frame], c)
                if self.RNN[layer].ispool:
                    h = h + self.ShortCut[(layer + 1) / 5 - 1](outForShortcut[(layer + 1) / 5 - 1][:,frame])
                    output_[:,frame] = self.RNN[layer].pool(h)
                else:
                    output_[:, frame] = h

                if layer == self.RNN_layer - 1:
                    output_[:, frame] = self.dropout2(output_[:, frame])
                else:
                    output_[:, frame] = output_[:, frame]
            output = output_

        out_action = Variable(torch.from_numpy(np.zeros((x.shape[0], x.shape[1], self.num_action)))).cuda().float()
        for i in range(x.shape[1]):
            out_action[:, i] = self.classifier(output[:, i].contiguous().view(x.shape[0], -1))

        if self.test_scheme == 1:
            ################# Scheme1 #####################
            out = torch.mean(out_action, 1)

            ################# Scheme 2 ###############################
        elif self.test_scheme == 2:
            if self.training:
                out = torch.mean(out_action,1)
            else:
                cls = []
                for i in range(out_action.shape[1]):
                    cls.append(torch.max(out_action[:,i],1)[1])
                cls = torch.stack(cls, dim=1)
                out = []
                for i in cls:
                    counts = np.bincount(i)
                    res = np.argmax(counts)
                    out.append(torch.zeros(self.num_action))
                    out[-1][res] = 1
                out = torch.stack(out, dim = 0).cuda()

        elif self.test_scheme == 3:
            ################# Scheme 3 #############################
            if self.training:
                out = torch.mean(out_action,1)
            else:
                out = out_action[:,-1]
        else:
            print("Wrong test_scheme")

        return out, out_action

def actionModel(num_action, batch_norm=False, pretrained=True, dropout=[0, 0, 0] , RNN_layer=15, test_scheme=1, detachout=0.0):

    net_dict = []
    block0 = [{'conv1_1': [3, 64, 3, 1, 1]}, {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1': [2, 2, 0]},
              {'conv2_1': [64, 128, 3, 1, 1]}, {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2': [2, 2, 0]},
              {'conv3_1': [128, 256, 3, 1, 1]}, {'conv3_2': [256, 256, 3, 1, 1]}, {'conv3_3': [256, 256, 3, 1, 1]},{'conv3_4': [256, 256, 3, 1, 1]}, {'pool3': [2, 2, 0]},
              {'conv4_1': [256, 512, 3, 1, 1]}, {'conv4_2': [512, 512, 3, 1, 1]}, {'conv4_3': [512, 512, 3, 1, 1]},{'conv4_4': [512, 512, 3, 1, 1]}, {'pool4': [2, 2, 0]},
              {'conv5_1': [512, 512, 3, 1, 1]}, {'conv5_2': [512, 512, 3, 1, 1]}, {'conv5_3': [512, 512, 3, 1, 1]},{'conv5_4': [512, 512, 3, 1, 1]}, {'pool5': [2, 2, 0]}]
    net_dict.append(block0)  # Basic VGG - net_dict[0]

    channels = 512

    model = SNN(net_dict, batch_norm, num_action, dropout=dropout, channels=channels, RNN_layer=RNN_layer, test_scheme=test_scheme, detachout=detachout)

    if pretrained:
        parameter_num = 16
        if batch_norm:
            vgg19 = models.vgg19_bn()
            vgg19.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'))
            parameter_num *= 6
        else:
            vgg19 = models.vgg19(pretrained=True)
            parameter_num *= 2
        vgg19_state_dict = vgg19.state_dict()
        vgg19_keys = vgg19_state_dict.keys()

        model_dict = model.state_dict()
        from collections import OrderedDict
        weights_load = OrderedDict()
        for i in range(parameter_num):
            weights_load[model.state_dict().keys()[i]] = vgg19_state_dict[vgg19_keys[i]]
        model_dict.update(weights_load)
        model.load_state_dict(model_dict)
        print("LOAD VGG19")

    return model

