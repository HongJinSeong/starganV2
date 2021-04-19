import os
from os.path import join as ospj
import time
import datetime
from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import build_model
from core.checkpoint import CheckpointIO

from core.data_loader import glob

from core.data_loader import InputFetcher     ### ORIGIN 버전

from core.data_loader import DS_for_train 
from core.data_loader import DS_for_valid 
from core.data_loader import DS_for_inference ###  inference용

import core.utils as utils
from metrics.eval import calculate_metrics

import json
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from torch.backends import cudnn
from metrics.fid import *
import pandas as pd

device = 'cuda' if torch.cuda.is_available else 'cpu'



class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        #self.ckptios = [CheckpointIO(ospj(args.pretrained_dir, '060000_nets_ema.ckpt'), **self.nets_ema)]  # base output 체크
        self.ckptios = [CheckpointIO(ospj(args.pretrained_dir, '496nets_ema.ckpt'), **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)


    def _load_checkpoint(self, step, load_optims=True):
        for i, ckptio in enumerate(self.ckptios):
            if load_optims == False and i == 2:
                print('optim not load')
            else:
                ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()


    def inference(self,src,ref,arr):
        ds=DS_for_inference(src,ref,self.args.img_size,arr)
        dataloader = DataLoader(ds, batch_size=self.args.batch_size, shuffle=False)

        self._load_checkpoint(self.args.resume_iter, False)
        val_out_ds=[]
        for i,ds in enumerate(dataloader):
            x_real = ds[0].to(device)
            y_org = ds[1].to(device)
            label = ds[2].to(device)
            nets_ema = self.nets_ema

            if not os.path.isdir(self.args.infer_outdir):
                os.makedirs(self.args.infer_outdir)
            val_out_ds.append(self.args.infer_outdir + str(i) + '.png')
            utils.translate_using_reference(nets_ema, self.args, x_real,None ,y_org, label,
                                           self.args.infer_outdir+str(i)+'.png')
        fid = calculate_fid_given_paths([ref, val_out_ds], img_size=512)
        print(fid)



def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    if args.mode=='inference':
        solver = Solver(args)
        cls_by_folder = glob('dataset/', '*.csv', True)

        arr = pd.read_csv('dataset/dataindex_00.csv', dtype='unicode')
        arr = arr.loc[:, ['sex', 'path']]

        for file_info in cls_by_folder:
            arrr = pd.read_csv(file_info, dtype='unicode')
            arrr = arrr.loc[:, ['sex', 'path']]
            arr=pd.concat([arr,arrr],axis=0)
        arr.drop_duplicates()
        src=np.load(args.infer_src)
        ref=np.load(args.infer_ref)
        solver.inference(src,ref,arr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=2,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=2,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=0,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')

    ## 기존 pretrain batchsize =5
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size for validation')

    ## learning rate 내림 (기존 1e-4 batch size 5일때 기준이기 때문)
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate for D, E and G')
    ## learning rate 내림 (기존 1e-6 batch size 5일때 기준이기 때문)
    parser.add_argument('--f_lr', type=float, default=2e-7,
                        help='Learning rate for F')

    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')

    # misc
    parser.add_argument('--mode', type=str, default='inference',
                        choices=['train', 'sample', 'eval','inference'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='datasets/test/src',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='datasets/afhq/val',
                        help='Directory containing validation images')
    parser.add_argument('--test_img_dir', type=str, default='data/mqset',
                        help='Directory containing test images')
    parser.add_argument('--sample_dir', type=str, default='outputs/hair/trainoutput',
                        help='Directory for saving generated images')

    ### train때는 pretrain checkpoint를 보게함
    parser.add_argument('--pretrained_dir', type=str, default='outputs/hair/checkpoints',
                        help='Directory for pretrained checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/afhq/checkpoints/',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--load_pretrained', type=str, default=True,
                        help='load pretrained or load checkpoint')

    parser.add_argument('--dataset_dir', type=str, default='imagelists',
                        help='Directory of train, valid image lists (npy files)')



    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval/k-hairstyle',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results/k-hairstyle',
                        help='Directory for saving generated images')
    parser.add_argument('--src_dir', type=str, default='sample_images/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='sample_images/ref',
                        help='Directory containing input reference images')

    parser.add_argument('--src_domain', type=int, default=0,
                        help='Source domain (e.g., 0, 1)')
    parser.add_argument('--trg_domain', type=int, default=1,
                        help='Target domain (e.g., 0, 1)')
    parser.add_argument('--num_sample', type=int, default=300,
                        help='Number of samples to generate')

    # face alignment (not used in k-hairstyle baseline)
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=30000)

    #####inference 파라미터###############
    parser.add_argument('--infer_src', type=str, default='samples_src_P.npy',
                        help='inference 소스 이미지 lists (npy files)')
    parser.add_argument('--infer_ref', type=str, default='samples_ref_P.npy',
                        help='inference용 레퍼런스 이미지 lists (npy files)')
    parser.add_argument('--infer_outdir', type=str, default='test_image/',
                        help='Directory of train, valid image lists (npy files)')

    args = parser.parse_args()
    main(args)
