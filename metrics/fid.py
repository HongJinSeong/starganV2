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

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
from core.data_loader import get_fid_loader
from core.data_loader import DS_for_FID

from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        ## 시간절약용
        inception = models.inception_v3(pretrained=False)
        inception.load_state_dict(torch.load('checkpoints/inception_v3_google-1a9a5a14.pth'))
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    fake_len = len(paths[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    #loaders = []
    #loaders.append(get_fid_loader(paths[0], img_size, batch_size= batch_size, trg_domain=trg_domain, fake_len = fake_len, mode = 'real',dataset_dir = dataset_dir))
    #loaders.append(get_fid_loader(paths[1], img_size, batch_size= batch_size, trg_domain=trg_domain, fake_len = fake_len, mode = 'fake',dataset_dir = dataset_dir))
    F_ds=DS_for_FID(paths[0],paths[1],img_size)
    dataloader = DataLoader(F_ds, batch_size=2, shuffle=False)
    actvsls=[]
    actvsls2=[]
    for i,datas in enumerate(dataloader):
        actvs = inception(datas[0].to(device))
        actvsls.append(actvs)

        actvs2 = inception(datas[1].to(device))
        actvsls2.append(actvs2)
        if i==1:
            break
    actvs = torch.cat(actvsls, dim=0).cpu().detach().numpy()
    mu1=(np.mean(actvs, axis=0))
    cov1=np.cov(actvs, rowvar=False)
    actvs2 = torch.cat(actvsls2, dim=0).cpu().detach().numpy()
    mu2 = (np.mean(actvs2, axis=0))
    cov2 = np.cov(actvs2, rowvar=False)
    fid_value = frechet_distance(mu1, cov1, mu2, cov2)
    return fid_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs=2, help='paths to real and fake images')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    args = parser.parse_args()
    fid_value = calculate_fid_given_paths(args.paths, args.img_size, args.batch_size)
    print('FID: ', fid_value)

# python -m metrics.fid --paths PATH_REAL PATH_FAKE