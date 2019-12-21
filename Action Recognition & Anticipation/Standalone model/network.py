import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from sync_batchnorm import SynchronizedBatchNorm2d
import random


class DRNN(nn.Module):
    def __init__(self, batch_norm, num_action, dropout, TD_rate, test_scheme=1, img_size=112, syn_bn=False):

        super(DRNN, self).__init__()

        # Configs
        self.channels = [3, 64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        self.RNN_layer = len(self.channels) - 1
        self.input_size = [img_size, img_size/4, img_size/4, img_size/4, img_size/4, img_size/4, img_size/8, img_size/8, img_size/8, img_size/8, img_size/16, img_size/16, img_size/16, img_size/16, img_size/32, img_size/32, img_size/32]
        self.out_size = [img_size/4, img_size/4, img_size/4, img_size/4, img_size/4, img_size/8, img_size/8, img_size/8, img_size/8, img_size/16, img_size/16, img_size/16, img_size/16, img_size/32, img_size/32, img_size/32, img_size/32]
        self.stride = [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
        self.shortCutLayer =   [   1, 3, 5, 7, 9,  11, 13, 15]
        self.mergeShortCutLayer = [2, 4, 6, 8, 10, 12, 14, 16]
        self.not_selftrans = [6, 10, 14]

        # Modules
        self.dropout_RNN = nn.Dropout2d(p=dropout[0])

        self.RNN = nn.ModuleList([self.make_RNNCell((self.channels[i]), (self.channels[i+1] + self.channels[i]), self.channels[i+1], detachout=TD_rate, kernel_size=7 if i == 0 else 3, stride=self.stride[i], syn_bn=syn_bn, pool=(i == 0), padding=3 if i == 0 else 1) for i in range(self.RNN_layer)])
        self.init_weight(self.RNN, xavier_gain=3.0)

        block = [[{'convsc_1': [self.channels[5], self.channels[7], 1, 2, 0]}], [{'convsc_2': [self.channels[9], self.channels[11], 1, 2, 0]}],
                 [{'convsc_3': [self.channels[13], self.channels[15], 1, 2, 0]}]]
        self.ShortCut = nn.ModuleList([self._make_layer(block[i], batch_norm, syn_bn=syn_bn) for i in range(len(block))])
        self.init_weight(self.ShortCut)

        self.classifier = nn.Sequential(nn.Linear(int(self.channels[15] * pow(self.out_size[-1],2)), int(self.channels[15]*self.out_size[-1])), nn.ReLU(inplace=True), nn.Dropout(p=dropout[1]), nn.Linear(int(self.channels[15]*self.out_size[-1]), num_action))#nn.Softmax(dim=1))
        self.sftmx = nn.LogSoftmax(dim=1)
        self.init_weight(self.classifier)

        # Parameters
        self.syn_bn = syn_bn
        self.num_action = num_action
        self.test_scheme = test_scheme

    def init_weight(self, model, xavier_gain=1.0):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=xavier_gain)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, net_dict, batch_norm=False, syn_bn=False):
        layers = []
        length = len(net_dict)
        for i in range(length):
            one_layer = net_dict[i]
            key = list(one_layer.keys())[0]
            v = one_layer[key]

            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                if batch_norm:
                    if syn_bn:
                        layers += [conv2d, SynchronizedBatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def make_RNNCell(self, in_channel1, in_channel2, out_channel, detachout, kernel_size, stride, padding, pool=False, syn_bn=False):
        class RNN_cell(nn.Module):
            def __init__(self, in_channel1, in_channel2, out_channel, pool, detachout, kernel_size, stride, padding):
                super(RNN_cell, self).__init__()
                self.outchannel = out_channel
                conv_data = nn.Conv2d(in_channels=in_channel1, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
                conv_ctrl = nn.Conv2d(in_channels=in_channel2, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=True)
                self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

                if syn_bn:
                    layers_data = [conv_data, SynchronizedBatchNorm2d(out_channel), torch.nn.ReLU()]
                    layers_ctrl = [conv_ctrl, SynchronizedBatchNorm2d(out_channel), torch.nn.Sigmoid()]
                else:
                    layers_data = [conv_data, nn.BatchNorm2d(out_channel), torch.nn.ReLU()]
                    layers_ctrl = [conv_ctrl, nn.BatchNorm2d(out_channel), torch.nn.Sigmoid()]

                self.conv_data = nn.Sequential(*layers_data)
                self.conv_ctrl = nn.Sequential(*layers_ctrl)

                self.ispool = pool
                self.detachout = detachout
                self.stride = stride

            def forward(self, x, c):
                input_data = x
                rand = random.random()
                if rand < self.detachout:
                    x_ctrl = x.detach()
                else:
                    x_ctrl = x

                input_ctrl = torch.cat((x_ctrl, c), 1)

                data = self.conv_data(input_data)
                ctrl = self.conv_ctrl(input_ctrl)

                if self.stride == 1:
                    output = data * ctrl
                else:
                    output = data * self.pool(ctrl)

                return output, ctrl

        return RNN_cell(in_channel1, in_channel2, out_channel, pool, detachout, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, initial):
        out_action = Variable(torch.zeros((x.shape[0], x.shape[1], self.num_action))).cuda().float()
        for frame in range(x.shape[1]):
            # 0
            out, initial[0] = self.RNN[0](x[:, frame], initial[0])
            out = self.RNN[0].pool(out)

            # 1
            short = out
            out, initial[1] = self.RNN[1](out, initial[1])
            out, initial[2] = self.RNN[2](out, initial[2])
            out = out + short

            # 3
            short = out
            out, initial[3] = self.RNN[3](out, initial[3])
            out, initial[4] = self.RNN[4](out, initial[4])
            out = out + short

            # 5
            short = out
            out, initial[5] = self.RNN[5](out, initial[5])
            out, initial[6] = self.RNN[6](out, initial[6])
            out = out + self.ShortCut[0](short)

            # 7
            short = out
            out, initial[7] = self.RNN[7](out, initial[7])
            out, initial[8] = self.RNN[8](out, initial[8])
            out = out + short

            # 9
            out, initial[9] = self.RNN[9](out, initial[9])
            out, initial[10] = self.RNN[10](out, initial[10])
            out = out + self.ShortCut[1](short)

            out = self.dropout_RNN(out)

            # 11
            short = out
            out, initial[11] = self.RNN[11](out, initial[11])
            out, initial[12] = self.RNN[12](out, initial[12])
            out = out + short

            # 13
            short = out
            out, initial[13] = self.RNN[13](out, initial[13])
            out, initial[14] = self.RNN[14](out, initial[14])
            out = out + self.ShortCut[2](short)

            # 15
            short = out
            out, initial[15] = self.RNN[15](out, initial[15])
            out, initial[16] = self.RNN[16](out, initial[16])
            out = out + short

            out = self.dropout_RNN(out)

            out_action[:, frame] = self.classifier(out.contiguous().view(x.shape[0], -1))
            if self.training:
                out_action[:, frame] = self.sftmx(2 * out_action[:, frame])
            else:
                out_action[:, frame] = self.sftmx(out_action[:, frame])

        if self.test_scheme == 1:
            out = torch.mean(out_action, 1)

        elif self.test_scheme == 2:
            if self.training:
                out = torch.mean(out_action, 1)
            else:
                cls = []
                for i in range(out_action.shape[1]):
                    cls.append(torch.max(out_action[:, i], 1)[1])
                cls = torch.stack(cls, dim=1)
                out = []
                for i in cls:
                    counts = np.bincount(i)
                    res = np.argmax(counts)
                    out.append(torch.zeros(self.num_action))
                    out[-1][res] = 1
                out = torch.stack(out, dim=0).cuda()

        elif self.test_scheme == 3:
            if self.training:
                out = torch.mean(out_action[:, out_action.shape[1]/2:out_action.shape[1]],1)
            else:
                out = torch.mean(out_action, 1)

        elif self.test_scheme == 4:
            total_len = out_action.shape[1]
            slice = total_len / 5
            out = Variable(torch.zeros((x.shape[0], slice, self.num_action))).cuda().float()
            for s in range(slice):
                out[:, s] = torch.mean(out_action[:, 5*s:5*(s+1)], 1)
            cls = []
            for i in range(out.shape[1]):
                cls.append(torch.max(out[:, i], 1)[1])
            cls = torch.stack(cls, dim=1)
            out_ = []
            for i in cls:
                counts = np.bincount(i)
                res = np.argmax(counts)
                out_.append(torch.zeros(self.num_action))
                out_[-1][res] = 1
            out = torch.stack(out_, dim=0).cuda()

        else:
            print("Wrong test_scheme")

        for i in range(len(initial)):
            initial[i] = initial[i].detach()
        return out, out_action, initial


def actionModel(num_action, batch_norm=False, dropout=[0, 0], test_scheme=1, TD_rate=0.5, image_size=112, syn_bn=False):

    model = DRNN(batch_norm, num_action, dropout=dropout, test_scheme=test_scheme, TD_rate=TD_rate, img_size=image_size, syn_bn=syn_bn)

    return model


