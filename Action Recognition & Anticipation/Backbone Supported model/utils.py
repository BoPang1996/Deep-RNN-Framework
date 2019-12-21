import math
import numpy as np

class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):

    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr

def IoU(bbox_predict, bbox_label):
    x1 = bbox_predict[0]
    y1 = bbox_predict[1]
    width1 = bbox_predict[2]
    height1 = bbox_predict[3]

    x2 = bbox_label[0]
    y2 = bbox_label[1]
    width2 = bbox_label[2]
    height2 = bbox_label[3]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    # return IOU
    return ratio


def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)

        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))

        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)

    assert x.shape == orig_shape
    return x


def mAP(predict, label, pre_pro):
    predict = np.array(predict)
    label = np.array(label)
    if len(predict) == 0:
        return None
    if pre_pro == 'sm':
        predict = softmax(predict)
    if pre_pro == 'Lsm':
        predict = np.power(math.e, predict)
    action_n = len(predict[0])
    APs = []
    for action in range(action_n):
        ap = AP(predict[:,action],label[:], action)
        if ap != None:
            APs.append(ap)
    APs = np.array(APs)
    return np.mean(APs)

def AP(_predicts, labels, action):
    score = []
    num_hits = 0.0

    predicts = -np.sort(-_predicts)
    idxs = np.argsort(-_predicts)
    for i, predict in enumerate(predicts):
        label = labels[idxs[i]]
        if int(label) == action:
            num_hits += 1.0
            score.append(num_hits / (i + 1.0))
    maxx = 0
    final = 0
    for i in range(len(score)):
        if score[len(score)-1-i] > maxx:
            maxx = score[len(score)-1-i]
        final += max(maxx, score[len(score)-1-i])

    if num_hits == 0:
        return None

    return final / num_hits

def accuracy(predict, label, pre_pro):
    predict = np.array(predict)
    label = np.array(label)
    if len(predict) == 0:
        return None
    if pre_pro == 'sm':
        predict = softmax(predict)
    if pre_pro == 'Lsm':
        predict = np.power(math.e, predict)
    total = len(predict)
    true = 0
    for i in range(total):
        result = np.argmax(predict[i])
        if result == label[i]:
            true += 1
    return float(true) / float(total)

