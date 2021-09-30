import os
import pathlib as pb
import argparse as ap
from shutil import copyfile

import cv2 as cv
from loguru import logger
import numpy as np



def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv.Laplacian(image, cv.CV_64F).var()


def parse_args():
    # construct the argument parse and parse the arguments
    ap_ = ap.ArgumentParser()
    ap_.add_argument("-i", "--images_folder", required=True, help="path to input directory of images")
    ap_.add_argument("-o", "--output_folder", required=True, help="path to output directory of images")
    ap_.add_argument("-t", "--threshold", type=float, default=100.0,
        help="focus measures that fall below this value will be considered 'blurry'")
    ap_.add_argument("-c", "--count", default=np.inf, type=int, help="path to input directory of images")
    ap_.add_argument('--soft-link', action='store_true', help='Use symbolic links.')
    args = ap_.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    image_paths = sorted(pb.Path(args.images_folder).rglob(r'*.png'))
    len_image_paths = len(image_paths)

    counter = 0
    for ind, ip in enumerate(image_paths):

        if counter >= args.count:
            break

        img = cv.imread(str(ip))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        var = variance_of_laplacian(img)

        if var > args.threshold:
            logger.success(f'{ind}/{len_image_paths}: {ip.name}: Not blurry ({var:.2f})!')

            src = str(ip)
            dst = str(pb.Path(args.output_folder) / ip.name)

            if args.soft_link:
                os.symlink(src, dst)
            else:
                copyfile(src, dst)

            counter += 1
        else:
            logger.error(f'{ind}/{len_image_paths}: {ip.name}: Blurry ({var:.2f})!')


if __name__ == '__main__':
    main()
