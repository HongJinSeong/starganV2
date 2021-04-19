"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse
import numpy as np
from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader, get_val_loader
from core.solver import Solver
from core.data_loader import glob

from torchvision.datasets import  *
import numpy as np
import os

import shutil
import inference as inf

def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    if args.mode=='inference':
        solver = inf.Solver(args)
    else:
        solver = Solver(args)
    if args.mode == 'train':
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers,
                                             dataset_dir=args.dataset_dir
                                             ),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers,
                                             dataset_dir=args.dataset_dir
                                             ),
                        val=get_val_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            dataset_dir=args.dataset_dir
                                            )
                        )
        #dataset 불러오기 ===> train
        cls_by_folder = glob('datasets/hair_pretrain/', '*', True)
        clslist=[]    ## 전체 구분 array
        Tds=[]        ## train 전체 data array
        ds_by_cls=[]  ## cls별 train array (ref image1의 생성물과 ref image2의 생성물간에 cycle consistency loss 계산해야되는데 같은 클래스에서 들고와야하기 때문에)

        # random 한 10000개
        for i,fol in enumerate(cls_by_folder):
            folsplit=fol.split('\\')
            clslist.append(folsplit[-1])
            ds = glob(fol, '*', True)
            Tds+=ds
            ds_by_cls.append(ds)

        Vds=[]
        Vds1 = []  ## validation 전체 data array 여자
        Vds2 = []  ## validation 전체 data array 남자
        cls_by_valfolder = glob('datasets/hair_preval/', '*', True)
        for i,fol in enumerate(cls_by_valfolder):
            ds = glob(fol, '*', True)
            if i == 0:
                Vds1 += ds
            else:
                Vds2 += ds

        solver.train(loaders,clslist,Tds,ds_by_cls,Vds1,Vds2)
    elif args.mode == 'sample':
        solver.sample()
    elif args.mode == 'eval':
        fid_values, fid_mean = solver.evaluate()
        for key, value in fid_values.items():
            print(key, value)
    elif args.mode=='inference':
        src=np.load(args.infer_src)
        ref=np.load(args.infer_ref)

        solver.inference(src,ref)

    else:
        raise NotImplementedError


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

    ### train때는 pretrain checkpoint를 보게하고
    ### tset시에는 실제 내가 저장한 checkpoint 보도록 해야함
    parser.add_argument('--pretrained_dir', type=str, default='outputs/hair/checkpoints/',
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

    #####inference 파라미터
    parser.add_argument('--infer_src', type=str, default='samples_src.npy',
                        help='inference 소스 이미지 lists (npy files)')
    parser.add_argument('--infer_ref', type=str, default='samples_ref.npy',
                        help='inference용 레퍼런스 이미지 lists (npy files)')
    parser.add_argument('--infer_outdir', type=str, default='outputs/SAOUTPUT/',
                        help='Directory of train, valid image lists (npy files)')

    args = parser.parse_args()
    main(args)
