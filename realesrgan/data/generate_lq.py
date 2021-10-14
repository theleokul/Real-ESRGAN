import os
import sys
import pathlib as pb
import argparse as ap

import torch
import yaml
from basicsr.utils.options import ordered_yaml, set_random_seed
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher

DIR_PATH = pb.Path(__file__).resolve().parent
ROOT_PATH = DIR_PATH.parent.parent
sys.path.append(str(DIR_PATH))
from realesrgan_dataset import RealESRGANDataset


def parse_args():
    ap_ = ap.ArgumentParser()
    ap_.add_argument("--hq", required=True, help="path to hq directory of images")
    ap_.add_argument("-o", "--output_folder", required=True, help="path to lq directory of images")
    ap_.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    ap_.add_argument("--random-seed", default=42, type=int, help="Random seed for the degradation pipeline.")
    args = ap_.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    opt['root_path'] = ROOT_PATH

    set_random_seed(args.random_seed)
    torch.use_deterministic_algorithms(True)

    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = os.path.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = os.path.expanduser(dataset['dataroot_lq'])

    dataset_opt = opt['datasets']['train']
    dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
    train_set = build_dataset(dataset_opt)
    train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
    train_loader = build_dataloader(
        train_set,
        dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=train_sampler,
        seed=args.random_seed
    )

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    train_sampler.set_epoch(0)
    prefetcher.reset()
    data = prefetcher.next()
    device = torch.device('cuda')

    while data is not None:
        # model.feed_data(train_data)

        gt = data['gt'].to(device)
        kernel1 = data['kernel1'].to(self.device)
        kernel2 = data['kernel2'].to(self.device)
        sinc_kernel = data['sinc_kernel'].to(self.device)



        train_data = prefetcher.next()


