import math
from os import read
import pathlib as pb
from tqdm import tqdm

import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import torchvision as tv
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg as sp_linalg

# from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.utils.registry import METRIC_REGISTRY



@METRIC_REGISTRY.register()
class FIDISMetric(nn.Module):
    """
    Calculates FID and IS
    """

    def __init__(self, real_features_npy_path=None):
        super().__init__()
        if real_features_npy_path is not None:
            self.real_features = np.load(real_features_npy_path)
        else:
            self.real_features = None

        self.inception_v3 = tv.models.inception_v3(pretrained=True)
        self.inception_v3.eval()

        for p in self.inception_v3.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.device('cuda'))

        # Preprocess data
        x = F.interpolate(x, size=(299, 299), mode='bilinear')
        x = (x - 0.5) * 2

        # N x 3 x 299 x 299
        x = self.inception_v3.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_v3.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_v3.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception_v3.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception_v3.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_v3.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception_v3.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception_v3.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception_v3.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception_v3.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception_v3.Mixed_7c(x)
        # Adaptive average pooling
        x = self.inception_v3.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception_v3.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)

        return x

    @torch.no_grad()
    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        # N x 2048
        x = self.inception_v3.fc(x)
        # N x 1000 (num_classes)
        x = F.softmax(x, dim=1)

        return x

    def calc_data(self, fake_inputs: list, real_inputs: list=None):
        output = {}

        if real_inputs is not None:
            output['real_features'] = self.calc_features(real_inputs)

        fake_features, fake_probs = self.calc_features_probs(fake_inputs)

        output['fake_features'] = fake_features
        output['fake_probs'] = fake_probs

        return output

    def calc_features(self, inputs: list):
        features = []
        for inputs_batch in tqdm(inputs):
            features_batch = self._features(inputs_batch)
            features.append(features_batch.detach().cpu().numpy())
        features = np.concatenate(features)
        return features

    def calc_features_probs(self, inputs: list):
        features = []
        probs = []

        for inputs_batch in tqdm(inputs):
            features_batch = self._features(inputs_batch)
            features.append(features_batch.detach().cpu().numpy())

            probs_batch = self._classifier(features_batch)
            probs.append(probs_batch.detach().cpu().numpy())

        features = np.concatenate(features)
        probs = np.concatenate(probs)

        return features, probs

    @staticmethod
    def calc_fid(real_features, fake_features):
        real_mu, real_cov = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        fake_mu, fake_cov = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

        diff_mu = np.sum((real_mu - fake_mu)**2)
        sq_prod_cov = sp_linalg.sqrtm(real_cov.dot(fake_cov))

        if np.iscomplexobj(sq_prod_cov):
            sq_prod_cov = sq_prod_cov.real

        fid = diff_mu + np.trace(real_cov + fake_cov - 2. * sq_prod_cov)

        return fid

    @staticmethod
    def calc_is(fake_probs):
        p_yx = fake_probs
        p_y = fake_probs.mean(axis=0, keepdims=True)
        kl_d = p_yx * (np.log(p_yx + 1e-8) - np.log(p_y + 1e-8))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = sum_kl_d.mean()
        is_ = np.exp(avg_kl_d)
        return is_

    def forward(self, fake_images: list, real_images: list=None) -> torch.Tensor:
        if real_images is None:
            output = self._forward1(fake_images)
        else:
            output = self._forward2(fake_images, real_images)
        return output

    def _forward1(self, fake_images: list) -> torch.Tensor:
        assert self.real_features is not None
        output = self.calc_data(fake_images)
        real_features = self.real_features
        fake_features, fake_probs = output['fake_features'], output['fake_probs']

        fid = self.calc_fid(real_features, fake_features)
        inception_score = self.calc_is(fake_probs)

        return fid, inception_score

    def _forward2(self, fake_images: list, real_images: list) -> torch.Tensor:
        output = self.calc_data(fake_images, real_images)
        real_features, fake_features, fake_probs = output['real_features'], output['fake_features'], output['fake_probs']

        fid = self.calc_fid(real_features, fake_features)
        inception_score = self.calc_is(fake_probs)

        return fid, inception_score


if __name__ == '__main__':
    path_to_real_images = '/mnt/sdb1/datasets/FFHQ_1024_70k/val/hq_usm_crop256_from_crop400lapvar50'
    save_real_features_to = '/mnt/sdb1/datasets/FFHQ_1024_70k/val/hq_usm_crop256_from_crop400lapvar50_features.npy'

    img_paths = pb.Path(path_to_real_images).rglob('*.png')

    for img_path in img_paths:
        img_path = str(img_path)

    fid = FIDISMetric().to(torch.device('cuda'))

    dataset = datasets.ImageFolder(path_to_real_images, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, num_workers=2, batch_size=1, shuffle=False)

    real_inputs = []
    for batch in dataloader:
        x, _ = batch
        real_inputs.append(x.to(torch.device('cuda')))

    real_features = np.load('/mnt/sdb1/datasets/FFHQ_1024_70k/val/hq_usm_crop256_from_crop400lapvar50_features.npy')
    fid.real_features = real_features
    real_features = fid.calc_features(real_inputs)
    score = fid(real_inputs)

    print(score)


    # np.save(save_real_features_to, real_features)
