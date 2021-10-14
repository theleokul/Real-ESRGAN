import math

import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import torchvision as tv
import numpy as np
from scipy import linalg as sp_linalg

from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.archs.vgg_arch import VGGFeatureExtractor



@LOSS_REGISTRY.register()
class PerceptualContextualLoss(nn.Module):
    """Similar to basicsr's PerceptualLoss, but with Contextual Loss component."""

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 contextual_weight=1.0,
                 band_width = 0.5,  # Used in contextual loss
                 criterion='l1'):

        super(PerceptualContextualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.contextual_weight = contextual_weight
        self.band_width = band_width
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt, input=None):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
            input (Tensor): Input to the model (LR) with sahpe (n, c, h, w)
        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        input_features = self.vgg(input) if input is not None else None  # Used in contextual loss

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        # calculate contextual loss
        if self.contextual_weight > 0:
            contextual_loss = 0
            for k in x_features.keys():
                x = x_features[k]
                y = gt_features[k]
                inp = input_features[k] if input_features is not None else None

                # For GT
                dist_raw = self._compute_cosine_distance(x, y)
                dist_tilde = self._compute_relative_distance(dist_raw)
                contextual = self._compute_contextual(dist_tilde, self.band_width)
                contextual = torch.mean(torch.max(contextual, dim=1)[0], dim=1)  # Eq(1)
                contextual_loss += torch.mean(-torch.log(contextual + 1e-5)) * self.layer_weights[k]  # Eq(5)

                # For input
                if inp is not None:
                    dist_raw = self._compute_cosine_distance(inp, x)
                    dist_tilde = self._compute_relative_distance(dist_raw)
                    contextual = self._compute_contextual(dist_tilde, self.band_width * 0.5)
                    contextual = torch.mean(torch.max(contextual, dim=1)[0], dim=1)  # Eq(1)
                    contextual_loss += torch.mean(-torch.log(contextual + 1e-5)) * self.layer_weights[k]  # Eq(5)

            contextual_loss *= self.contextual_weight
        else:
            contextual_loss = None

        return percep_loss, style_loss, contextual_loss

    def _compute_cosine_distance(self, x, y):
        # mean shifting by channel-wise mean of `y`.
        y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
        x_centered = x - y_mu
        y_centered = y - y_mu

        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)

        # channel-wise vectorization
        N, C, *_ = x.size()
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

        # consine similarity
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

        # convert to distance
        dist = 1 - cosine_sim

        return dist

    def _compute_relative_distance(self, dist_raw):
        dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
        dist_tilde = dist_raw / (dist_min + 1e-5)
        return dist_tilde

    def _compute_contextual(self, dist_tilde, band_width):
        w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
        cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
        return cx

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class BCEWithLogitsLoss(nn.Module):

    def __init__(self, loss_weight=1.0, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.bce_wlogits_loss = nn.BCEWithLogitsLoss(**kwargs)
        self.loss_weight = loss_weight

    def forward(self, pred, gt):
        return self.bce_wlogits_loss(pred, gt) * self.loss_weight


@LOSS_REGISTRY.register()
class GANFeatureMatchingLoss(nn.Module):

    def __init__(self,
                 layer_weights,
                 loss_weight=1.0,
                 criterion='l1',
                 apply_gram=False):

        super(GANFeatureMatchingLoss, self).__init__()

        self.loss_weight = loss_weight
        self.layer_weights = layer_weights
        self.apply_gram = apply_gram

        if self.apply_gram:
            print('##: Gram is used in GANFeatureMatchingLoss')

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """

        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, dis_outputs, gt_dis_outputs):
        if self.loss_weight > 0:
            gfm_loss = 0
            for k in self.layer_weights.keys():
                feats = dis_outputs[k]
                gt_feats = gt_dis_outputs[k]

                if self.apply_gram:
                    feats = self._gram_mat(feats)
                    gt_feats = self._gram_mat(gt_feats)

                if self.criterion_type == 'fro':
                    gfm_loss += torch.norm(feats - gt_feats, p='fro') * self.layer_weights[k]
                else:
                    gfm_loss += self.criterion(feats, gt_feats) * self.layer_weights[k]

            # gfm_loss /= len(self.layer_weights)
            gfm_loss *= self.loss_weight
        else:
            gfm_loss = None

        return gfm_loss
