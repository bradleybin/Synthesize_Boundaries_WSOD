#!/usr/bin/python3
#coding=utf-8

import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image,image1, mask, gt,mask1,attention):
        for op in self.ops:
            image,image1, mask, gt,mask1,attention = op(image, image1,mask, gt,mask1,attention)
        return image,image1, mask, gt, mask1,attention

class RGBDCompose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, depth, mask,mask1,attention):
        for op in self.ops:
            image, depth, mask, mask1,attention = op(image, depth, mask,mask1,attention)
        return image, depth, mask,mask1,attention


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image,image1, mask, gt, mask1,attention):
        image = (image - self.mean)/self.std
        image1 = (image1 - self.mean) / self.std
        # mask /= 255
        return image, image1, mask, gt, mask1,attention

class random_rotate(object):
    def __call__(self, x,y,z, gt, mask1,attention):
        flip_flag = np.random.randint(0, 2)
        #print(flip_flag)
        if flip_flag == 1:
            angle = np.random.randint(-15,15)
            #print('x',x.shape)
            h, w, c = x.shape
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            x = cv2.warpAffine(x, M, (w, h))
            y = cv2.warpAffine(y, M, (w, h))
            z = cv2.warpAffine(z, M, (w, h))
            gt = cv2.warpAffine(gt, M, (w, h))
            mask1 = cv2.warpAffine(mask1, M, (w, h))
            attention = cv2.warpAffine(attention, M, (w, h))

        return x, y, z, gt, mask1,attention

# class Resize(object):
#     def __init__(self, H, W):
#         self.H = H
#         self.W = W
#
#     def __call__(self, image, image1, mask, gt, mask1,attention):
#         image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         image1 = cv2.resize(image1, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)
#         gt = cv2.resize(gt, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)
#         mask1 = cv2.resize(mask1, dsize=(self.W, self.H), interpolation=cv2.INTER_NEAREST)
#         attention = cv2.resize(attention, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
#         return image, image1, mask, gt, mask1,attention

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, image1, mask, gt, mask1, attention):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        image1 = cv2.resize(image1, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask1 = cv2.resize(mask1, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        attention = cv2.resize(attention, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, image1, mask, gt, mask1, attention

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, image1, mask, gt, mask1,attention):
        H,W,_ = image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        image1 = image1[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask  = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        gt = gt[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask1 = mask1[ymin:ymin + self.H, xmin:xmin + self.W, :]
        attention = attention[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return image, image1, mask, gt, mask1,attention

class RandomHorizontalFlip(object):
    def __call__(self, image, image1, mask, gt, mask1,attention):
        if np.random.randint(2)==1:
            image = image[:,::-1,:].copy()
            image1 = image1[:, ::-1, :].copy()
            mask  =  mask[:,::-1,:].copy()
            gt = gt[:, ::-1, :].copy()
            mask1 = mask1[:, ::-1, :].copy()
            attention = attention[:, ::-1, :].copy()
        return image, image1, mask, gt, mask1, attention

class ToTensor(object):
    def __call__(self, image, image1, mask, gt, mask1,attention):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image1 = torch.from_numpy(image1)
        image1 = image1.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)
        gt  = torch.from_numpy(gt)
        gt  = gt.permute(2, 0, 1)
        mask1 = torch.from_numpy(mask1)
        mask1 = mask1.permute(2, 0, 1)
        attention = torch.from_numpy(attention)
        attention = attention.permute(2, 0, 1)
        return image, image1, mask.mean(dim=0, keepdim=True), gt.mean(dim=0, keepdim=True), mask1.mean(dim=0, keepdim=True), attention.mean(dim=0, keepdim=True)

