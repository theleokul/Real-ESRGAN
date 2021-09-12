import argparse
import cv2
import numpy as np
import os
import sys
from basicsr.utils import scandir
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
from pathlib import Path

from skimage.transform import rescale, resize, downscale_local_mean



def main(opt):
    output_lq = os.path.join(opt.output, 'lq')
    output_hq = os.path.join(opt.output, 'hq')

    os.makedirs(output_lq, exist_ok=True)
    os.makedirs(output_hq, exist_ok=True)

    files = list(scandir(opt.input, full_path=True))

    for file in tqdm(files):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[0:2]

        h_hq = h - h % 4
        w_hq = w - w % 4

        img_lq = cv2.resize(img, (w_hq // 4, h_hq // 4), interpolation=cv2.INTER_LINEAR)
        img_hq = cv2.resize(img, (w_hq, h_hq), interpolation=cv2.INTER_LINEAR)

        out_lq = os.path.join(output_lq, Path(file).name)
        out_hq = os.path.join(output_hq, Path(file).name)

        cv2.imwrite(out_lq, img_lq)
        cv2.imwrite(out_hq, img_hq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='D:/datasets/FFHQ_wild_human_cuts/val_hq', help='Input folder')
    parser.add_argument('--output', type=str, default='D:/datasets/FFHQ_wild_human_cuts/val', help='Output folder')
    args = parser.parse_args()

    main(args)
