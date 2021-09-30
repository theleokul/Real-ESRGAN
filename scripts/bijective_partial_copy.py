import os
import pathlib as pb
from shutil import copyfile

from tqdm import tqdm


HQ_ANCHOR = '/mnt/sdb1/datasets/FFHQ_wild_human_cuts/val_hq_lapvar_100_count_100'
HQ = '/mnt/sdb1/datasets/FFHQ_wild_human_cuts/predict_val_lq_wlapfilterandmasks_20000_tile_512_tilepad_256'
OUTPUT = '/mnt/sdb1/datasets/FFHQ_wild_human_cuts/predict_val_lq_wlapfilterandmasks_20000_tile_512_tilepad_256_count_16'
LIM = 16


def main():
    anchors = os.listdir(HQ_ANCHOR)

    os.makedirs(OUTPUT, exist_ok=True)

    for i, anchor in tqdm(enumerate(anchors)):

        if i >= LIM:
            break

        # -8 -- remove _out.png
        # gt_name = pred[:-8] + '.png'
        gt_name = anchor[:-4] + '_out.png'
        # gt_name = anchor

        gt_src = os.path.join(HQ, gt_name)
        gt_dest = os.path.join(OUTPUT, gt_name)

        copyfile(gt_src, gt_dest)


if __name__ == '__main__':
    main()
