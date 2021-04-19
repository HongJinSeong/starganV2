"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
######### freeze D 와 label smoothing 적용 해보기
######### 해보고 안좋으면 feature matching loss도 적용해보기
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

from core.data_loader import InputFetcher     ### ORIGIN 버전
from core.data_loader import DS_for_train ### 내가 짜던거
from core.data_loader import DS_for_valid ### 내가 짜던거

import core.utils as utils
from metrics.eval import calculate_metrics

import json
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from metrics.fid import *
from torch.utils.tensorboard import SummaryWriter
from inference import *

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

device = 'cuda' if torch.cuda.is_available else 'cpu'
torch.manual_seed(0)
if device == 'cuda':
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

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

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                base_optimizer=torch.optim.Adam
                optimizer=SAM(self.nets[net].parameters(), base_optimizer, lr=args.f_lr if net == 'mapping_network' else args.lr,
                              betas=[args.beta1, args.beta2],
                              weight_decay=args.weight_decay)
                self.optims[net] = optimizer

            ## pretrain checkpoin load를 위함

            ### 496이 최저점이기 때문에 496살려서 checkpoint 로 사용
            if args.load_pretrained==True:
                self.ckptios = [
                    CheckpointIO(args.pretrained_dir+'496nets.ckpt', **self.nets),
                    CheckpointIO(args.pretrained_dir+'496nets_ema.ckpt', **self.nets_ema),
                    CheckpointIO(args.pretrained_dir+'496nets_optims.ckpt', **self.optims)]
            else:
                self.ckptios = [
                    CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets.ckpt'), **self.nets),
                    CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema),
                    CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_optims.ckpt'), **self.optims)]

        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_nets_ema.ckpt'), **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)


    # 학습시 checkpoint를 구분하여 저장
    def _save_checkpoint(self, step):
        for i,ckptio in enumerate(self.ckptios):
            name=''
            if i==0:
                name=step+'nets.ckpt'
            if i==1:
                name=step+'nets_ema.ckpt'
            if i==2:
                name=step+ 'nets_optims.ckpt'
            ckptio.save(name)

    def _load_checkpoint(self, step,load_optims=True):
        for i,ckptio in enumerate(self.ckptios):
            if load_optims==False and i==2:
                print('optim not load')
            else:
                ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders,clslist,Tds,ds_by_cls,Vds1,Vds2):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        train_ds=DS_for_train(clslist,Tds,ds_by_cls,args)
        valid_ds = DS_for_valid(clslist, Vds1,Vds2,  args)

        # log (tensorboard)
        writer = SummaryWriter('outputs/hair_SAM/summaries')

        dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

        Vdataloader= DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

        # resume training if necessary
        #if args.resume_iter > 0:
        self._load_checkpoint(args.resume_iter,False)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds

        print('Start training...')
        start_time = time.time()
        pidx=0
        for epoch in range(args.resume_iter, args.total_iters):
            for i,ds in enumerate(dataloader):
                x_real=ds[0].to(device)
                y_org=ds[1].to(device)
                x_ref=ds[2].to(device)
                x_ref2=ds[3].to(device)
                y_trg=ds[4].to(device)
                z_trg=ds[5].to(device)
                z_trg2=ds[6].to(device)
                real_path = ds[7]
                ref_path = ds[8]
                ref2_path = ds[9]
                cls = ds[10]
                cls_ref = ds[11]

                masks = None

                # train the discriminator
                d_loss, d_losses_latent = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.first_step(zero_grad=True)
                compute_d_loss(
                    nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)[0].backward()
                optims.discriminator.second_step(zero_grad=True)

                d_loss, d_losses_ref = compute_d_loss(
                    nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.first_step(zero_grad=True)
                compute_d_loss(
                    nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)[0].backward()
                optims.discriminator.second_step(zero_grad=True)

                # train the generator
                g_loss, g_losses_latent = compute_g_loss(
                    nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
                self._reset_grad()
                g_loss.backward()
                optims.generator.first_step(zero_grad=True)
                optims.mapping_network.first_step(zero_grad=True)
                optims.style_encoder.first_step(zero_grad=True)
                compute_g_loss(
                    nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)[0].backward()
                optims.generator.second_step(zero_grad=True)
                optims.mapping_network.second_step(zero_grad=True)
                optims.style_encoder.second_step(zero_grad=True)

                g_loss, g_losses_ref = compute_g_loss(
                    nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
                self._reset_grad()
                g_loss.backward()
                optims.generator.first_step(zero_grad=True)
                compute_g_loss(
                    nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)[0].backward()
                optims.generator.second_step(zero_grad=True)

                # compute moving average of network parameters
                moving_average(nets.generator, nets_ema.generator, beta=0.999)
                moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
                moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

                # decay weight for diversity sensitive loss
                if args.lambda_ds > 0:
                        args.lambda_ds -= (initial_lambda_ds / args.ds_iter)

                    # print out log info
                if i==0 or (i + 1) % args.sample_every == 0:
                        elapsed = time.time() - start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                        log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, train_ds.__len__())
                        all_losses = dict()
                        for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                                ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                            for key, value in loss.items():
                                all_losses[prefix + key] = value

                                writer.add_scalar(key,
                                                  value,
                                                  pidx)
                        all_losses['G/lambda_ds'] = args.lambda_ds
                        log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                        print(log)

                    # test generate images

                if  i==0 or (i + 1) % args.sample_every == 0:
                        os.makedirs(args.sample_dir, exist_ok=True)
                        utils.debug_image_H(nets_ema, args, x_real, y_org,x_ref, y_trg , step=pidx)

                    # save model checkpoints
                if  i==0 or (i + 1) % args.sample_every == 0:
                        self._save_checkpoint('outputs/hair_SAM/checkpoints/'+str(pidx))


                        val_out_ds=[]
                    # compute FID and LPIPS if necessary
                if  i==0 or (i + 1) % args.sample_every == 0:
                        for v_idx,ds in enumerate(Vdataloader):
                            x_real = ds[0].to(device)
                            y_org = ds[1].to(device)
                            x_ref = ds[2].to(device)
                            y_trg = ds[3].to(device)
                            z_trg = ds[4].to(device)
                            real_path = ds[5]
                            ref_path = ds[6]
                            cls = ds[7]
                            cls_ref = ds[8]
                            os.makedirs('outputs/hair_SAM/validoutput/'+str(pidx), exist_ok=True)
                            utils.translate_using_reference(nets_ema, args, x_real,None, x_ref, y_trg,'outputs/hair_SAM/validoutput/'+str(pidx)+'/'+real_path[0].split('\\')[-1] )
                            val_out_ds.append(str('outputs/hair_SAM/validoutput/'+str(pidx)+'/'+real_path[0].split('\\')[-1]))

                        fid=calculate_fid_given_paths([Vds1,val_out_ds],img_size=512)
                        print(fid)
                        writer.add_scalar('Frechet Inception Distance',
                                          fid,
                                          pidx)
                        pidx+=1



    def sample(self,):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(ospj(args.result_dir, str(args.trg_domain)), exist_ok=True)
        self._load_checkpoint(args.resume_iter)
        trg_domain = str(args.trg_domain)
        src_root, ref_root = os.path.join(args.src_dir,trg_domain), os.path.join(args.ref_dir,trg_domain)
        src_list = os.listdir(src_root)
        ref_list = os.listdir(ref_root)
        src_list.sort()
        ref_list.sort()
        print("Src len: ",len(src_list),"Trg len: ",len(ref_list))
        total_len = len(src_list)
        transform = transforms.Compose([
            transforms.Resize([args.img_size, args.img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_cnt = 0
        for i in tqdm(range(len(src_list))):
            label = torch.LongTensor([int(args.trg_domain)])
            img = Image.open(os.path.join(src_root, src_list[i])).convert('RGB')
            img2 = Image.open(os.path.join(ref_root, ref_list[i])).convert('RGB')
            src, trg  = transform(img), transform(img2)
            masks = None
            img_cnt = utils.translate_using_reference_sample(nets_ema, args, src, trg, masks, label,
                                                    img_cnt, total_len, img_name = src_list[i])
            if img_cnt >= total_len:
                break

    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        fid_values, fid_mean = calculate_metrics(nets_ema, args, step=resume_iter, mode='test')
        return fid_values, fid_mean


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)


    loss_real = adv_loss(out, 1)

    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out,1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=None) # originally fake image mask
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc
    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like the other optimizers, you should first call `first_step` and the `second_step`; see the documentation for more info.")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm