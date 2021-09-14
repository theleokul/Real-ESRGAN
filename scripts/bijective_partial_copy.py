import os
import pathlib as pb
from shutil import copyfile


GT_HQ = 'D:\\datasets\\FFHQ_wild_human_cuts\\val\\lq'
GEN_HQ = '../results_0'
OUTPUT_GT = '../lq'



def main():
    preds = os.listdir(GEN_HQ)

    os.makedirs(OUTPUT_GT, exist_ok=True)

    for pred in preds:
        gt_name = pred[:-8] + '.png'
        gt_src = os.path.join(GT_HQ, gt_name)
        gt_dest = os.path.join(OUTPUT_GT, gt_name)

        copyfile(gt_src, gt_dest)


if __name__ == '__main__':
    main()
