import os
import pathlib as pb
from shutil import copyfile

from tqdm import tqdm

FROM_DIR_PATH = '/mnt/sdb1/datasets/predict_test_set_wlapfilterandmasks_gfm+disinbottleneck_26000_tile_256_tilepad_10_outscale_2/'
TO_DIR_PATH = '/mnt/sdb1/datasets/predict_test_set_together_20211006/'
ANCHOR_DIR_PATH = '/mnt/sdb1/datasets/predict_test_set_remini/'
# ANCHOR_DIR_PATH = None
POSTFIX = '_gfm+disinbottleneck'
STEM_TRUNC = 4
USE_SOFT_LINK = False



def main():
    os.makedirs(TO_DIR_PATH, exist_ok=True)

    if ANCHOR_DIR_PATH is not None:
        anchor_stems = os.listdir(ANCHOR_DIR_PATH)
        anchor_stems = [pb.Path(n).stem for n in anchor_stems]

    from_paths = os.listdir(FROM_DIR_PATH)
    for fp in tqdm(from_paths):
        fp = pb.Path(FROM_DIR_PATH) / fp

        stem = fp.stem
        if STEM_TRUNC > 0:
            stem = stem[:-STEM_TRUNC]

        if ANCHOR_DIR_PATH is not None and stem in anchor_stems:
            tp = pb.Path(TO_DIR_PATH) / (stem + POSTFIX + fp.suffix)

            if USE_SOFT_LINK:
                os.symlink(str(fp), str(tp))
            else:
                copyfile(str(fp), str(tp))
        else:
            tp = pb.Path(TO_DIR_PATH) / (stem + POSTFIX + fp.suffix)

            if USE_SOFT_LINK:
                os.symlink(str(fp), str(tp))
            else:
                copyfile(str(fp), str(tp))


if __name__ == '__main__':
    main()
