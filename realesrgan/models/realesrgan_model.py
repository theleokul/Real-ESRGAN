import os.path as osp
import pathlib as pb
import sys
from collections import defaultdict

import numpy as np
import random
import torch
from tqdm import tqdm
from basicsr.utils import imwrite, tensor2img
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric
from basicsr.losses import build_loss
from collections import OrderedDict
from torch.nn import functional as F
from cleanfid import fid
from loguru import logger as loguru_logger
from basicsr.utils.registry import METRIC_REGISTRY

try:
    from focal_frequency_loss import FocalFrequencyLoss as FFL
except:
    pass



@MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel):
    """RealESRGAN Model"""

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt.get('queue_size', 180)

    def init_training_settings(self):
        # NOTE: Overriding just to add GAN Feature Matching Loss
        super(RealESRGANModel, self).init_training_settings()

        train_opt = self.opt['train']
        if train_opt.get('gfm_opt'):
            self.cri_gfm = build_loss(train_opt['gfm_opt']).to(self.device)
        else:
            self.cri_gfm = None

        if train_opt.get('ffl_opt'):
            self.cri_ffl = FFL(
                loss_weight=train_opt['gfm_opt'].get('loss_weight', 1.),
                alpha=train_opt['gfm_opt'].get('alpha', 1.)
            )  # initialize nn.Module class
        else:
            self.cri_ffl = None

        if train_opt.get('fmss_opt'):
            # Face mask segmentation subtask
            self.cri_fmss = build_loss(train_opt['fmss_opt']).to(self.device)
        else:
            self.cri_fmss = None

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # training pair pool
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, 'queue size should be divisible by batch size'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            if data.get('mask_gt', None) is not None and data.get('face_mask_gt', None) is not None:
                self.mask_gt = data['mask_gt'].to(self.device)[:, None]  # Add color channel as well
                self.face_mask_gt = data['face_mask_gt'].to(self.device)[:, None]
                (self.gt, self.gt_usm, self.mask_gt, self.face_mask_gt), self.lq = paired_random_crop(
                    [self.gt, self.gt_usm, self.mask_gt, self.face_mask_gt],
                    self.lq, gt_size,
                    self.opt['scale']
                )
            elif data.get('mask_gt', None) is not None:
                self.mask_gt = data['mask_gt'].to(self.device)[:, None]  # Add color channel as well
                (self.gt, self.gt_usm, self.mask_gt), self.lq = paired_random_crop(
                    [self.gt, self.gt_usm, self.mask_gt],
                    self.lq, gt_size,
                    self.opt['scale']
                )
            elif data.get('face_mask_gt', None) is not None:
                self.face_mask_gt = data['face_mask_gt'].to(self.device)[:, None]
                (self.gt, self.gt_usm, self.face_mask_gt), self.lq = paired_random_crop(
                    [self.gt, self.gt_usm, self.face_mask_gt],
                    self.lq, gt_size,
                    self.opt['scale']
                )
            else:
                (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                      self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

            if data.get('mask_gt', None) is not None:
                self.mask_gt = data['mask_gt'].to(self.device)[:, None]  # Add color channel as well
            if data.get('face_mask_gt', None) is not None:
                self.face_mask_gt = data['face_mask_gt'].to(self.device)[:, None]  # Add color channel as well

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        # super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)

        ##################

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        # Parse FID opt
        if with_metrics:
            fid_meta_info = self.opt['val']['metrics'].get('fid', None)
            fid_real_features = fid_meta_info['real_features'] if fid_meta_info is not None else None
            with_fid = fid_meta_info is not None

            if not hasattr(self, 'fidis'):
                self.fidis = METRIC_REGISTRY.get('FIDISMetric')(fid_real_features)

            is_meta_info = self.opt['val']['metrics'].get('is', None)
            with_is = is_meta_info is not None
        else:
            fid_meta_info = fid_real_features = None
            with_fid = with_is = False

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)

        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        fid_preds = []
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            if with_fid:
                fid_preds.append(visuals['result'])

            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val'].get('suffix', None) is not None:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{self.opt["name"]}.png')
                        # save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name in ['fid', 'is']:
                        continue
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            if with_fid:
                # NOTE: To inference fidis, transfer inception to gpu for a short time
                torch.cuda.empty_cache()
                self.fidis = self.fidis.to(torch.device('cuda'))
                self.metric_results['fid'], is_score = self.fidis(fid_preds)

                if with_is:
                    self.metric_results['is'] = is_score

                self.fidis = self.fidis.to(torch.device('cpu'))

            for metric in self.metric_results.keys():
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        ##################

        self.is_train = True

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        #####################  Just for generation of validation dataset. Do not forget to turn off shuffling
        # Save lq, hq
        # visuals = self.get_current_visuals()
        # lq_img = tensor2img([visuals['lq']])
        # gt_img = tensor2img([visuals['gt']])
        # gt_usm_img = tensor2img([self.gt_usm.detach().cpu()])

        # save_lq_img_path = osp.join(self.opt['path']['lq'], f'{current_iter:06d}.png')
        # imwrite(lq_img, save_lq_img_path)

        # save_hq_img_path = osp.join(self.opt['path']['hq'], f'{current_iter:06d}.png')
        # imwrite(gt_img, save_hq_img_path)

        # save_hq_usm_img_path = osp.join(self.opt['path']['hq_usm'], f'{current_iter:06d}.png')
        # imwrite(gt_usm_img, save_hq_usm_img_path)
        #####################

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                if hasattr(self, 'mask_gt'):
                    mask_gt = self.mask_gt
                    l_g_pix = self.cri_pix(self.output * mask_gt, l1_gt * mask_gt)
                else:
                    l_g_pix = self.cri_pix(self.output, l1_gt)

                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                if hasattr(self, 'mask_gt'):
                    mask_gt = self.mask_gt
                    perceptual_losses = self.cri_perceptual(self.output * mask_gt, percep_gt * mask_gt)
                else:
                    perceptual_losses = self.cri_perceptual(self.output, percep_gt)

                l_g_percep = l_g_style = l_g_contextual = None
                if len(perceptual_losses) == 2:
                    l_g_percep, l_g_style = perceptual_losses
                elif len(perceptual_losses) == 3:
                    l_g_percep, l_g_style, l_g_contextual = perceptual_losses

                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
                if l_g_contextual is not None:
                    l_g_total += l_g_contextual
                    loss_dict['l_g_contextual'] = l_g_contextual

            if self.cri_ffl:
                if hasattr(self, 'mask_gt'):
                    mask_gt = self.mask_gt
                    l_g_ffl = self.cri_ffl(self.output * mask_gt, l1_gt * mask_gt)
                else:
                    l_g_ffl = self.cri_ffl(self.output, l1_gt)

                l_g_total += l_g_ffl
                loss_dict['l_g_ffl'] = l_g_ffl

            # gan loss
            dis_in_bottleneck = self.opt['network_d'].get('dis_in_bottleneck', False)
            with_face_mask = hasattr(self, 'face_mask_gt')

            gfm_layer_name_list = list(self.cri_gfm.layer_weights.keys()) if self.cri_gfm else []
            if with_face_mask:
                fake_g_pred = self.net_d(
                    self.output
                    , face_mask=self.face_mask_gt
                    , layer_name_list=gfm_layer_name_list
                )
            else:
                fake_g_pred = self.net_d(
                    self.output
                    , layer_name_list=gfm_layer_name_list
                )

            fake_g_pred_output = fake_g_pred['output'] if isinstance(fake_g_pred, dict) else fake_g_pred

            l_g_gan_local = self.cri_gan(
                fake_g_pred_output
                , True
                , is_disc=False
            )
            l_g_total += l_g_gan_local
            loss_dict['l_g_gan_local'] = l_g_gan_local

            if dis_in_bottleneck:
                fake_g_pred_bottleneck = fake_g_pred['bottleneck']
                l_g_gan_bottleneck = self.cri_gan(
                    fake_g_pred_bottleneck
                    , True
                    , is_disc=False
                )
                l_g_total += l_g_gan_bottleneck
                loss_dict['l_g_gan_bottleneck'] = l_g_gan_bottleneck

            # gfm loss
            if self.cri_gfm:
                if with_face_mask:
                    real_g_pred = self.net_d(
                        gan_gt
                        , face_mask=self.face_mask_gt
                        , layer_name_list=gfm_layer_name_list
                    )
                else:
                    real_g_pred = self.net_d(
                        gan_gt
                        , layer_name_list=gfm_layer_name_list
                    )
                l_g_gfm = self.cri_gfm(fake_g_pred, real_g_pred)
                l_g_total += l_g_gfm
                loss_dict['l_g_gfm'] = l_g_gfm

            if self.cri_fmss:
                # fake_g_pred['face_mask']: B, 1, H=256, W=256
                l_g_fmss_fake = self.cri_fmss(
                    fake_g_pred['face_mask'],
                    self.face_mask_gt
                )
                l_g_total += l_g_fmss_fake
                loss_dict['l_g_fmss_fake'] = l_g_fmss_fake

                if 'real_g_pred' in locals():
                    l_g_fmss_real = self.cri_fmss(
                        real_g_pred['face_mask'],
                        self.face_mask_gt
                    )
                    l_g_total += l_g_fmss_real
                    loss_dict['l_g_fmss_real'] = l_g_fmss_real

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()

        # real
        if with_face_mask:
            real_d_pred = self.net_d(gan_gt, face_mask=self.face_mask_gt)
        else:
            real_d_pred = self.net_d(gan_gt)
        real_d_pred_output = real_d_pred['output'] if isinstance(real_d_pred, dict) else real_d_pred

        l_d_real = 0
        l_d_real_local = self.cri_gan(
           real_d_pred_output
            , True
            , is_disc=True
        )
        loss_dict['l_d_real_local'] = l_d_real_local
        loss_dict['out_d_real_local'] = torch.mean(real_d_pred_output.detach())
        l_d_real += l_d_real_local

        if dis_in_bottleneck:
            l_d_real_bottleneck = self.cri_gan(
                real_d_pred['bottleneck']
                , True
                , is_disc=True
            )
            loss_dict['l_d_real_bottleneck'] = l_d_real_bottleneck
            loss_dict['out_d_real_bottleneck'] = torch.mean(real_d_pred['bottleneck'].detach())
            l_d_real += l_d_real_bottleneck

        if self.cri_fmss:
            l_d_fmss_real = self.cri_fmss(
                real_d_pred['face_mask'],
                self.face_mask_gt
            )
            l_d_real += l_d_fmss_real
            loss_dict['l_d_fmss_real'] = l_d_fmss_real

        l_d_real.backward()

        # fake
        if with_face_mask:
            fake_d_pred = self.net_d(self.output.detach().clone(), face_mask=self.face_mask_gt)  # clone for pt1.9
        else:
            fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        fake_d_pred_output = fake_d_pred['output'] if isinstance(fake_d_pred, dict) else fake_d_pred

        l_d_fake = 0
        l_d_fake_local = self.cri_gan(fake_d_pred_output, False, is_disc=True)
        loss_dict['l_d_fake_local'] = l_d_fake_local
        loss_dict['out_d_fake_local'] = torch.mean(fake_d_pred_output.detach())
        l_d_fake += l_d_fake_local

        if dis_in_bottleneck:
            l_d_fake_bottleneck = self.cri_gan(
                fake_d_pred['bottleneck']
                , True
                , is_disc=True
            )
            loss_dict['l_d_fake_bottleneck'] = l_d_fake_bottleneck
            loss_dict['out_d_fake_bottleneck'] = torch.mean(fake_d_pred['bottleneck'].detach())
            l_d_fake += l_d_fake_bottleneck

        if self.cri_fmss:
            l_d_fmss_fake = self.cri_fmss(
                fake_d_pred['face_mask'],
                self.face_mask_gt
            )
            l_d_fake += l_d_fmss_fake
            loss_dict['l_d_fmss_fake'] = l_d_fmss_fake

        l_d_fake.backward()

        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
