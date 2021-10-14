import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data


@DATASET_REGISTRY.register()
class RealESRGANDataset(data.Dataset):
    """
    Dataset used for Real-ESRGAN model.
    """

    def __init__(self, opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
                self.paths = [os.path.join(self.gt_folder, v) for v in paths]
        self.paths = sorted(self.paths)

        self.mask_paths = None
        if opt.get('meta_mask_info', None) is not None:
            with open(opt['meta_mask_info']) as fin:
                paths = [line.strip() for line in fin]
                self.mask_paths = [os.path.join(self.gt_folder, v) for v in paths]
            self.mask_paths = sorted(self.mask_paths)

        self.face_mask_paths = None
        if opt.get('meta_face_mask_info', None) is not None:
            with open(opt['meta_face_mask_info']) as fin:
                paths = [line.strip() for line in fin]
                self.face_mask_paths = [os.path.join(self.gt_folder, v) for v in paths]
            self.face_mask_paths = sorted(self.face_mask_paths)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        gt_mask_path = gt_face_mask_path = None
        if self.mask_paths is not None:
            gt_mask_path = self.mask_paths[index]
        if self.face_mask_paths is not None:
            gt_face_mask_path = self.face_mask_paths[index]

        # avoid errors caused by high latency in reading files
        retry = 3
        mask_bytes = face_mask_bytes = None
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                if gt_mask_path is not None:
                    mask_bytes = self.file_client.get(gt_mask_path, 'gt_mask')
                if gt_face_mask_path is not None:
                    face_mask_bytes = self.file_client.get(gt_face_mask_path, 'gt_face_mask')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                if self.mask_paths is not None:
                    gt_mask_path = self.mask_paths[index]
                if self.face_mask_paths is not None:
                    gt_face_mask_path = self.face_mask_paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            print(gt_path)
            raise Exception()

        mask_gt = None
        if mask_bytes is not None:
            mask_gt = imfrombytes(mask_bytes, float32=True)

        face_mask_gt = None
        if face_mask_bytes is not None:
            face_mask_gt = imfrombytes(face_mask_bytes, float32=True)
            # For consistency between face_mask_gt and mask_gt
            if mask_gt is not None:
                face_mask_gt *= mask_gt

        # -------------------- augmentation for training: flip, rotation -------------------- #
        if mask_gt is not None and face_mask_gt is not None:
            img_gt, mask_gt, face_mask_gt = augment([img_gt, mask_gt, face_mask_gt], self.opt['use_hflip'], self.opt['use_rot'])
        elif mask_gt is not None:
            img_gt, mask_gt = augment([img_gt, mask_gt], self.opt['use_hflip'], self.opt['use_rot'])
        elif face_mask_gt is not None:
            img_gt, face_mask_gt = augment([img_gt, face_mask_gt], self.opt['use_hflip'], self.opt['use_rot'])
        else:
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        if mask_gt is not None:
            assert img_gt.shape == mask_gt.shape, 'Images and masks are not consistent.'
            mask_gt = mask_gt[..., 0]  # Remove color channel as mask is binary
        if face_mask_gt is not None:
            assert img_gt.shape == face_mask_gt.shape, 'Images and masks are not consistent.'
            if face_mask_gt.ndim == 3:
                face_mask_gt = face_mask_gt[..., 0]  # Remove color channel as mask is binary

        # crop or pad to 400: 400 is hard-coded. You may change it accordingly
        crop_pad_size = 400
        # crop_pad_size = self.opt['gt_size']

        # pad
        h, w = img_gt.shape[0:2]
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)

            # NOTE: In that case we do not use mask as image is smaller than crop
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            if mask_gt is not None:
                mask_gt = cv2.copyMakeBorder(mask_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            if face_mask_gt is not None:
                face_mask_gt = cv2.copyMakeBorder(face_mask_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]

            if mask_gt is not None and face_mask_gt is not None:
                sample_face = bool(np.random.randint(2))  # 0 - silhoette, 1 - face
                sample_mask = face_mask_gt if sample_face else mask_gt
                top, left = self.gen_top_left_from_mask(sample_mask, h, w, crop_pad_size)
                mask_gt = mask_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
                face_mask_gt = face_mask_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            elif mask_gt is not None:
                top, left = self.gen_top_left_from_mask(mask_gt, h, w, crop_pad_size)
                mask_gt = mask_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            elif face_mask_gt is not None:
                top, left = self.gen_top_left_from_mask(face_mask_gt, h, w, crop_pad_size)
                face_mask_gt = face_mask_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]
            else:
                # randomly choose top and left coordinates
                top = random.randint(0, h - crop_pad_size)
                left = random.randint(0, w - crop_pad_size)

            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

        if mask_gt is not None:
            mask_gt = torch.tensor(mask_gt, dtype=img_gt.dtype)
        if face_mask_gt is not None:
            face_mask_gt = torch.tensor(face_mask_gt, dtype=img_gt.dtype)

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {
            'gt': img_gt
            , 'kernel1': kernel
            , 'kernel2': kernel2
            , 'sinc_kernel': sinc_kernel
            , 'gt_path': gt_path
        }

        if mask_gt is not None:
            return_d['mask_gt'] = mask_gt
        if face_mask_gt is not None:
            return_d['face_mask_gt'] = face_mask_gt

        return return_d

    def gen_top_left_from_mask(self, mask_gt, h, w, crop_pad_size):
        # Choose randomly center point based on mask
        pos_centrs_y, pos_centrs_x = np.nonzero(mask_gt)
        pos_centrs = np.stack([pos_centrs_y, pos_centrs_x], axis=1)

        if pos_centrs.shape[0] > 0:
            centr = random.choice(pos_centrs)

            top = centr[0] - crop_pad_size // 2
            left = centr[1] - crop_pad_size // 2

            # Correct top, left in case when point is out of the borders
            top = np.min([top, h - crop_pad_size])  # Max range limit
            top = np.max([0, top])  # Min range limit
            left = np.min([left, w - crop_pad_size])  # Max range limit
            left = np.max([0, left])  # Min range limit
        else:
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)

        return top, left

    def __len__(self):
        return len(self.paths)
