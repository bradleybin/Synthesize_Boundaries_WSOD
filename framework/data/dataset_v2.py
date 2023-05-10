#!/usr/bin/python3
#coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset, DataLoader
import random

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs    = kwargs
        #print('\nParameters...')
        # for k, v in self.kwargs.items():
        #     print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean      = np.array([[[124.55, 118.90, 102.94]]])
            self.std       = np.array([[[ 56.77,  55.97,  57.50]]])
        elif 'DUT-OMRON' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.61, 121.86, 114.92]]])
            self.std       = np.array([[[ 58.10,  57.16,  61.09]]])
        elif 'MSRA-10K' in self.kwargs['datapath']:
            self.mean      = np.array([[[115.57, 110.48, 100.00]]])
            self.std       = np.array([[[ 57.55,  54.89,  55.30]]])
        elif 'MSRA-B' in self.kwargs['datapath']:
            self.mean      = np.array([[[114.87, 110.47,  95.76]]])
            self.std       = np.array([[[ 58.12,  55.30,  55.82]]])
        elif 'SED2' in self.kwargs['datapath']:
            self.mean      = np.array([[[126.34, 133.87, 133.72]]])
            self.std       = np.array([[[ 45.88,  45.59,  48.13]]])
        elif 'PASCAL-S' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.02, 112.75, 102.48]]])
            self.std       = np.array([[[ 59.81,  58.96,  60.44]]])
        elif 'HKU-IS' in self.kwargs['datapath']:
            self.mean      = np.array([[[123.58, 121.69, 104.22]]])
            self.std       = np.array([[[ 55.40,  53.55,  55.19]]])
        elif 'SOD' in self.kwargs['datapath']:
            self.mean      = np.array([[[109.91, 112.13,  93.90]]])
            self.std       = np.array([[[ 53.29,  50.45,  48.06]]])
        elif 'THUR15K' in self.kwargs['datapath']:
            self.mean      = np.array([[[122.60, 120.28, 104.46]]])
            self.std       = np.array([[[ 55.99,  55.39,  56.97]]])
        elif 'SOC' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.48, 111.78, 101.27]]])
            self.std       = np.array([[[ 58.51,  56.73,  56.38]]])
        else:
            #raise ValueError
            self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
            self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):

        self.video_type = 'data/DUTS'
        path = cfg.datapath + '/synthetic/21_img'
        img_lines = os.listdir(path)
        self.samples = []
        for line in img_lines:
                #imagepath = cfg.datapath + '/image/' + line.strip() + '.jpg'
                imagepath = cfg.datapath + '/image/' + line.strip()
                #maskpath  = cfg.datapath + '/scribble/'  + line.strip() + '.png'
                maskpath = cfg.datapath + '/scribble/' + line.strip().replace('.jpg','.png')


                gtpath = cfg.datapath + '/mask/' + line.strip().replace('.jpg', '.png')
                attentionpath = cfg.datapath + '/attention_map/' + line.strip().replace('.jpg', '_diff.png')

                imageinsertpath = []
                gtinsertpath = []
                for i in range(21,31):
                    imageinsertpath.append(cfg.datapath + '/synthetic/' + str(i) + '_img/' + line.strip())
                    gtinsertpath.append(cfg.datapath + '/synthetic/' + str(i) + '_label/' + line.strip().replace('.jpg', '.png'))
                #main_data_easy_noshadow_lessnear1


                if cfg.mode == 'test':
                    maskpath = cfg.datapath + '/mask/' + line.strip().replace('.jpg','.png')
                # print('imagepath',imagepath)
                # print('maskpath', maskpath)
                self.samples.append([imagepath, maskpath, gtpath, imageinsertpath, gtinsertpath,attentionpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(

                                                    transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.RandomHorizontalFlip(),
                                                    transform.RandomCrop(320, 320),
                                                    transform.random_rotate(),
                                                    transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath, gtpath, imageinsertpath, gtinsertpath,attentionpath = self.samples[idx]
        #print('imagepath!!!!!!!!!!!', imagepath)

        randomdata = random.randint(0, 9)
        try:
            imageinsert = cv2.imread(imageinsertpath[randomdata]).astype(np.float32)[:, :, ::-1]
            gtinsert = cv2.imread(gtinsertpath[randomdata]).astype(np.float32)[:, :, ::-1]
        except:
            print('不对经0')
            imageinsert = cv2.imread(imageinsertpath[0]).astype(np.float32)[:, :, ::-1]
            gtinsert = cv2.imread(gtinsertpath[0]).astype(np.float32)[:, :, ::-1]

        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        gt                  = cv2.imread(gtpath).astype(np.float32)[:, :, ::-1]
        attention           = cv2.imread(attentionpath).astype(np.float32)[:, :, ::-1]
        H, W, C             = mask.shape

        gtinsert = 2 * gtinsert / 255

        image, imageinsert, mask, gt, gtinsert, attention         = self.transform(image, imageinsert, mask, gt, gtinsert, attention)
        mask1 = gtinsert
        gtinsert = gtinsert + mask
        mask[mask == 0.] = 255.
        mask[mask == 2.] = 0.
        gtinsert[gtinsert == 0.] = 255.
        gtinsert[gtinsert == 2.] = 0.
        attention = attention / 255

        return image, imageinsert, mask, (H, W), maskpath.split('/')[-1], gt, gtinsert, mask1, attention



    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='./DUTS')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)
    batch_idx = -1
    image, mask = prefetcher.next()
    image = image[0].permute(1,2,0).cpu().numpy()*cfg.std + cfg.mean
    mask  = mask[0].cpu().numpy()
    plt.subplot(121)
    plt.imshow(np.uint8(image))
    plt.subplot(122)
    plt.imshow(mask)
    input()

