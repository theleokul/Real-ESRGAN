# flake8: noqa
import sys
import pathlib as pb
import os.path as osp
from basicsr.train import train_pipeline

ROOT_PATH = pb.Path(__file__).parent.parent
sys.path.append(str(ROOT_PATH))
import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
