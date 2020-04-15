import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable


class MarginLoss(nn.Module):
    """MarginLoss for Capsule Network
    Args:
                m_pos: A float number
                m_neg: A float number
                lambda_: A float number in [0, 1]
                prediction: Output of Network, a tensor of shape [batch, class_num]
                target: Label of classification, a tensor of shape [batch, ]
    Returns:
                A tensor of shape [1], mean loss of batch
    """
    def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def makeOneHot(self, prediction, target):
        temp = torch.zeros(prediction.size()).long()
        if target.is_cuda:
            temp = temp.cuda()
        temp = temp.scatter_(dim=1, index=target.data.view(-1, 1), 1)
        return Variable(temp)

    def forward(self, prediction, target):
        target = self.makeOneHot(prediction, target)
        punish_FP = target.float() * F.relu(self.m_pos - prediction).pow(2)
        punish_FN = self.lambda_ * (1 - target.float()) * F.relu(prediction - self.m_neg).pow(2)
        loss = punish_FP + punish_FN
        return loss.sum(dim=1).mean()



class SpreadLoss(nn.Module):
    """SpreadLoss
    Args:
                margin_: A float number
                prediction: Output of Network, a tensor of shape [batch, class_num]
                target: Label of classification, a tensor of shape [batch, ]
    Returns:
                gap_mit: A tensor of shape [1], mean loss of batch
    """
    def __init__(self, margin_=0.5):
        super(SpreadLoss, self).__init__()
        self.margin_ = margin_

    def makeOneHot(self, prediction, target):
        temp = torch.zeros(prediction.size()).long()
        if target.is_cuda:
            temp = temp.cuda()
        temp = temp.scatter_(dim=1, index=target.data.view(-1, 1), 1)
        return Variable(temp)

    def forward(self, prediction, target):
        target = self.makeOneHot(prediction, target)
        pred_shape = prediction.size()
        mask_t = (target == 1)
        mask_i = (target == 0)
        pred_t = torch.reshape(
            torch.masked_select(prediction, mask_t), shape=[pred_shape[0], 1])
        pred_i = torch.reshape(
            torch.masked_select(prediction, mask_i), shape=[pred_shape[0], pred_shape[1]-1])
        gap_mit = torch.sum(torch.relu(self.margin_ - (pred_t - pred_i)).pow(2))
        return gap_mit



class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss
    Args:
                prediction: Output of Network, a tensor of shape [batch, class_num], activated by softmax
                target: Label of classification, a tensor of shape [batch, ]
    Returns:
                Mean loss of batch
    Comments:
                Treat each sample equally
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def makeOneHot(self, prediction, target):
        temp = torch.zeros(prediction.size()).long()
        if target.is_cuda:
            temp = temp.cuda()
        temp = temp.scatter_(dim=1, index=target.data.view(-1, 1), 1)
        return Variable(temp)

    def forward(self, prediction, target):
        target = self.makeOneHot(prediction, target)
        loss = - (target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))
        return loss.sum(dim=1).mean()



class DSCLoss(nn.Module):
    """DSCLoss: Dice Loss for Data-imbalanced NLP Tasks (Multi-Classification)
    Args:
                smooth: A float number to smooth loss, and avoid NaN error, default: 1
                prediction: Output of Network, a tensor of shape [batch, class_num]
                target: Label of classification, a tensor of shape [batch, ]
    Returns:
                Loss tensor according to args reduction
    Comments:
                Suitable for imbalanced data.
    """
    def __init__(self, smooth=1):
        super(DSCLoss, self).__init__()
        self.smooth = smooth
    
    def makeOneHot(self, prediction, target):
        temp = torch.zeros(prediction.size()).long()
        if target.is_cuda:
            temp = temp.cuda()
        temp = temp.scatter_(dim=1, index=target.data.view(-1, 1), 1)
        return Variable(temp)

    def forward(self, prediction, target):
        target = self.makeOneHot(prediction, target)

        num = 2 * (1.0 - prediction) * prediction * target + self.smooth
        den = (1.0 - prediction) * prediction + target + self.smooth
        loss = torch.mean(1.0 - num / den, dim=0).sum()
        return loss / target.size(1)



class DiceLoss(nn.Module):
    """DiceLoss: A kind of Dice Loss (Multi-Classification)
    Args:
                smooth: A float number to smooth loss, and avoid NaN error, default: 1
                prediction: Output of Network, a tensor of shape [batch, class_num]
                target: Label of classification, a tensor of shape [batch, ]
    Returns:
                Loss tensor according to args reduction
    """
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def makeOneHot(self, prediction, target):
        temp = torch.zeros(prediction.size()).long()
        if target.is_cuda:
            temp = temp.cuda()
        temp = temp.scatter_(dim=1, index=target.data.view(-1, 1), 1)
        return Variable(temp)

    def forward(self, prediction, target):
        target = self.makeOneHot(prediction, target)

        num = 2 * prediction * target + self.smooth
        den = prediction.pow(2) + target.pow(2) + self.smooth
        loss = torch.mean(1.0 - num / den, dim=0).sum()
        return loss / target.size(1)



class TverskyLoss(nn.Module):
    """TverskyLoss: A kind of Dice loss (Multi-Classification)
    Args:
                alpha: A float number in [0, 1]
                beta: A float number in [0, 1]
                smooth: A float number to smooth loss, and avoid NaN error, default: 1
    Returns:
                Loss tensor according to args reduction
    """
    def __init__(self, alpha, beta, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def makeOneHot(self, prediction, target):
        temp = torch.zeros(prediction.size()).long()
        if target.is_cuda:
            temp = temp.cuda()
        temp = temp.scatter_(dim=1, index=target.data.view(-1, 1), 1)
        return Variable(temp)

    def forward(self, prediction, target):
        target = self.makeOneHot(prediction, target)

        num = prediction * target + self.smooth
        den = prediction * target + self.alpha * prediction * (1.0 - target) + self.beta * (1.0 - prediction) * target + self.smooth
        loss = torch.mean(1.0 - num / den, dim=0).sum()
        return loss / target.size(1)
        