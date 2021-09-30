import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY



@LOSS_REGISTRY.register()
class GANFeatureMatchingLoss(nn.Module):

    def __init__(self,
                 layer_weights,
                 loss_weight=1.0,
                 criterion='l1'):

        super(GANFeatureMatchingLoss, self).__init__()

        self.loss_weight = loss_weight
        self.layer_weights = layer_weights

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, dis_outputs, gt_dis_outputs):
        if self.loss_weight > 0:
            gfm_loss = 0
            for k in self.layer_weights.keys():
                if self.criterion_type == 'fro':
                    gfm_loss += torch.norm(dis_outputs[k] - gt_dis_outputs[k], p='fro') * self.layer_weights[k]
                else:
                    gfm_loss += self.criterion(dis_outputs[k], gt_dis_outputs[k]) * self.layer_weights[k]

            gfm_loss /= len(self.layer_weights)
            gfm_loss *= self.loss_weight
        else:
            gfm_loss = None

        return gfm_loss
