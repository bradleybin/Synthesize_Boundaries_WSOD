#!/usr/bin/python3
#coding=utf-8



import sys
import datetime
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader

from data import dataset_v2
from data import dataset_test
from network import Network
import logging as logger
from attention_lscloss import *
from lscloss import *
import numpy as np
from tools import *

from torch.autograd import Variable
import matplotlib.pyplot as plt
from visdom import Visdom
viz = Visdom()

TAG = "network"
SAVE_PATH = "model"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum


def get_polylr(base_lr, last_epoch, num_steps, power):
    return base_lr * (1.0 - min(last_epoch, num_steps-1) / num_steps) **power

def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b

class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)

train_path = 'data/DUTS'
datapath_test = 'data/DUTS-TE'
BASE_LR = 1e-5
MAX_LR = 5e-3
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.02}]
loss_lsc_kernels_desc_defaults_insert = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
batch = 16
l = 0.1
l_isnert = 0.3


def train(Dataset, Net):
    ## dataset
    cfg = Dataset.Config(datapath=train_path, savepath=SAVE_PATH, mode='train', batch=batch, lr=1e-3, momen=0.9, decay=5e-4, epoch=55)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## network
    net = Net(cfg)
    # print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))
    criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')
    iou_loss = IOU(size_average=True)
    loss_lsc = LocalSaliencyCoherence().cuda()
    net.train(True)
    net.train()
    net.cuda()
    criterion.cuda()
    iou_loss.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)

    global_step = 0
    db_size = len(loader)

    # -------------------------- training ------------------------------------
    for epoch in range(cfg.epoch):
        batch_idx = -1

        cnt = 0
        #mae = 0

        for i, data_batch in enumerate(loader):

            cnt = cnt + 1
            image, image1, mask, _, _, _, insert_mask, mask1, new_attention = data_batch
            image, image1, mask, insert_mask, mask1, new_attention = Variable(image.cuda()), \
                                               Variable(image1.cuda()), \
                                               Variable(mask.cuda()), \
                                                Variable(insert_mask.cuda()), \
                                                Variable(mask1.cuda()), \
                                                Variable(new_attention.cuda())
            image, image1, mask, insert_mask, mask1, new_attention = image.float(), image1.float(), mask.float(), insert_mask.float(), mask1.float(), new_attention.float()
            #insert_attention = new_attention * (1 - mask1 / 2)


            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            batch_idx += 1
            global_step += 1


            ######  saliency structure consistency loss  ######
            image_scale = F.interpolate(image, scale_factor=0.3, mode='bilinear', align_corners=True)
            out2, out3, out4, out5 = net(image, 'Train')
            out2_s, out3_s, out4_s, out5_s = net(image_scale, 'Train')
            out2_scale = F.interpolate(out2[:, 1:2], scale_factor=0.3, mode='bilinear', align_corners=True)
            loss_ssc = SaliencyStructureConsistency(out2_s[:, 1:2], out2_scale, 0.85)


            insert_out2, insert_out3, insert_out4, insert_out5 = net(image1, 'Train')
            ######  global consistency loss  ######


            #print('torch.max(mask1)', torch.max(mask1))
            B, C, HH, WW = out2[:, 1:2].shape
            #print('mask1mask1mask1', torch.max(mask1))
            out2_i = out2[:, 1:2] * (1-mask1/2)
            out2_reshape = out2_i.reshape(B, C, HH*WW)

            out2_reshape = out2_reshape.reshape(B, C, HH*WW)
            insert_out2_reshape = insert_out2[:, 1:2].reshape(B, C, HH*WW)
            global_loss0 = - F.cosine_similarity(out2_reshape, insert_out2_reshape.detach(), dim=-1).mean() -  F.cosine_similarity(insert_out2_reshape, out2_reshape.detach(), dim=-1).mean()
            #global_loss = 0.5 * SaliencyStructureConsistency(out2[:, 1:2], insert_out2[:, 1:2], 0.85) + 0.5 * global_loss0
            #global_loss_out2 = 0.5 * SaliencyStructureConsistency(out2[:, 1:2], insert_out2[:, 1:2], 0.5) + 0.5 * global_loss0
                               #+ 0.5\
                               #* iou_loss(out2[:, 1:2], insert_out2[:, 1:2])
            global_loss_out2 = 0.5 * SaliencyStructureConsistency_mse(out2_i, insert_out2[:, 1:2], 0.5) + 0.5 * global_loss0
            global_loss = global_loss_out2



            ######   label for partial cross-entropy loss  ######
            gt = mask.squeeze(1).long()
            bg_label = gt.clone()
            fg_label = gt.clone()
            bg_label[gt != 0] = 255
            fg_label[gt == 0] = 255

            gt1 = insert_mask.squeeze(1).long()
            insert_bg_label = gt1.clone()
            insert_bg_label[gt1 != 0] = 255


            ######################################################################################################################################
            image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
            sample = {'rgb': image_}
            out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults_insert, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss2 = loss_ssc + criterion(out2, fg_label) + criterion(out2, bg_label) + l_isnert * loss2_lsc  ## dominant loss


            ######  auxiliary losses  ######
            out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss3_lsc = loss_lsc(out3_, loss_lsc_kernels_desc_defaults_insert, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l_isnert * loss3_lsc
            out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss4_lsc = loss_lsc(out4_, loss_lsc_kernels_desc_defaults_insert, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l_isnert * loss4_lsc
            out5_ = F.interpolate(out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss5_lsc = loss_lsc(out5_, loss_lsc_kernels_desc_defaults_insert, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss5 = criterion(out5, fg_label) + criterion(out5, bg_label) + l_isnert * loss5_lsc

            ######################################################################################################################################
            org_loss = loss2 * 1 + loss3 * 0.8 + loss4 * 0.6 + loss5 * 0.4
            if epoch > 28:

                image1_ = F.interpolate(image1, scale_factor=0.25, mode='bilinear', align_corners=True)
                sample1 = {'rgb': image1_}
                insert_out2_ = F.interpolate(insert_out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
                insert_loss2_lsc = loss_lsc(insert_out2_, loss_lsc_kernels_desc_defaults_insert, loss_lsc_radius, sample1, image1_.shape[2], image1_.shape[3])['loss']
                insert_loss2 = loss_ssc + criterion(insert_out2, fg_label) + criterion(insert_out2, insert_bg_label) + l_isnert * insert_loss2_lsc  ## dominant loss

                ######  auxiliary losses  ######
                insert_out3_ = F.interpolate(insert_out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
                insert_loss3_lsc = loss_lsc(insert_out3_, loss_lsc_kernels_desc_defaults_insert, loss_lsc_radius, sample1, image1_.shape[2], image1_.shape[3])['loss']
                insert_loss3 = criterion(insert_out3, fg_label) + criterion(insert_out3, insert_bg_label) + l_isnert * insert_loss3_lsc
                insert_out4_ = F.interpolate(insert_out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
                insert_loss4_lsc = loss_lsc(insert_out4_, loss_lsc_kernels_desc_defaults_insert, loss_lsc_radius, sample1, image1_.shape[2], image1_.shape[3])['loss']
                insert_loss4 = criterion(insert_out4, fg_label) + criterion(insert_out4, insert_bg_label) + l_isnert * insert_loss4_lsc
                insert_out5_ = F.interpolate(insert_out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
                insert_loss5_lsc = loss_lsc(insert_out5_, loss_lsc_kernels_desc_defaults_insert, loss_lsc_radius, sample1, image1_.shape[2], image1_.shape[3])['loss']
                insert_loss5 = criterion(insert_out5, fg_label) + criterion(insert_out5, insert_bg_label) + l_isnert * insert_loss5_lsc
                insert_loss = insert_loss2*1 + insert_loss3*0.8 + insert_loss4*0.6 + insert_loss5*0.4
            ######################################################################################################################################
                ######  objective function  ######
                loss = org_loss + global_loss + insert_loss
            else:
                loss = org_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | org_loss=%.6f | insert_loss=%.6f | global_loss=%.6f' % (SAVE_PATH, datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), loss.item(), loss.item(), loss.item(), loss.item())
                print(msg)
                logger.info(msg)


            #image, mask = prefetcher.next()
        if epoch > 28:
            if (epoch+1) % 1 == 0 or (epoch+1) == cfg.epoch:
                torch.save(net.state_dict(), cfg.savepath+'/modelinsert/model2-'+str(epoch+1)+'.pt')
        #print('mae-----------------------------------------------: ', mae/cnt )

if __name__=='__main__':
    for i in range(4):
        train(dataset_v2, Network)

