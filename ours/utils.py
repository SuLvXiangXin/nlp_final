from torch.nn import CrossEntropyLoss
import torch
import numpy as np


class WeightedLoss:
    """ Weighted loss for diverse categories"""

    def __init__(self, w=(0.25, 1)):
        self.w = np.array(w) / np.sum(w)
        # use CrossEntropy as baseline
        self.loss_func = CrossEntropyLoss()

    def __call__(self, pred, target):
        unrelated = target == 3
        related = target != 3
        loss = 0
        if unrelated.sum():
            loss += self.w[0] * self.loss_func(pred[unrelated], target[unrelated])
        if related.sum():
            loss += self.w[1] * self.loss_func(pred[related], target[related])
        return loss


def score(pred, act):
    """ score the prediction """
    s = 0
    s += 0.25 * torch.logical_and(pred == act, act == 3).sum()
    s += 0.25 * torch.logical_and(act != 3, pred != 3).sum()
    s += 0.75 * torch.logical_and(pred == act, act != 3).sum()
    return s
