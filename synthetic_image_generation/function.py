import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import torch.nn.functional as F
import torch
import pylab
import scipy.misc
import imageio
import os
from skimage import morphology
import random
from skimage import measure
from skimage import morphology
import math





def FindForeBackPoint(fg_label, bg_label, num_bgwin, num_parameter, scale_img_LAB, mode): 
    skeleton_bg_label = morphology.skeletonize(bg_label)
    skeleton_fg_label = morphology.skeletonize(fg_label)
    H, W = skeleton_bg_label.shape
    bg_coordinate = []
    fg_coordinate = []

    for i in range(H):
        for j in range(W):
            if skeleton_bg_label[i, j] != 0:  
                bg_coordinate.append([i, j])
    for i in range(H):
        for j in range(W):
            if skeleton_fg_label[i, j] != 0: 
                fg_coordinate.append([i, j])
    if mode == 'random':
        bg_point_num = random.randint(int(len(bg_coordinate) * 1 / 5), int(len(bg_coordinate) * 4 / 5))
        fg_point_num = random.randint(int(len(fg_coordinate) * 1 / 3), int(len(fg_coordinate) * 2 / 3))
        try:
            bg_min_i, bg_min_j = bg_coordinate[bg_point_num]
            fg_min_i, fg_min_j = fg_coordinate[fg_point_num]
        except:
            bg_min_i, bg_min_j = bg_coordinate[0]
            fg_min_i, fg_min_j = fg_coordinate[0]

    if mode == 'near':
        min_dis = 1000
        bg_min_i = 0
        bg_min_j = 0
        fg_min_i = 0
        fg_min_j = 0
        for f in fg_coordinate:
            f1 = f[0]
            f2 = f[1]
            for b in bg_coordinate:
                b1 = b[0]
                b2 = b[1]
                distance = np.sqrt(np.square(f1 - b1) + np.square(f2 - b2))
                if distance < min_dis:
                    min_dis = distance
                    bg_min_i = b1
                    bg_min_j = b2
                    fg_min_i = f1
                    fg_min_j = f2

    if mode == 'bestbg':
        bg_min_var = 1
        bg_min_i = 0
        bg_min_j = 0
        for i in range(H):
            for j in range(W):
                if skeleton_bg_label[i, j] != 0: 
                    local_map = scale_img_LAB[np.max([(i - num_bgwin), 0]): np.min([(i + num_bgwin), (H - 1)]),
                                np.max([(j - num_bgwin), 0]): np.min([(j + num_bgwin), (W - 1)]), :]
                    r_h, r_w, _ = local_map.shape
                    local_map = local_map.reshape((r_h * r_w, 3))
                    var = np.mean(local_map.std(axis=0))
                    if var <= bg_min_var:  
                        bg_min_var = var
                        bg_min_i = i
                        bg_min_j = j
        fg_coordinate = []
        for i in range(H):
            for j in range(W):
                if skeleton_fg_label[i, j] != 0: 
                    fg_coordinate.append((i, j))

        fg_point_num = random.randint(int(len(fg_coordinate) * 1 / 3), int(len(fg_coordinate) * 2 / 3))
        fg_min_i, fg_min_j = fg_coordinate[fg_point_num]


    if mode == 'random' or mode == 'near':
        bg_min_var = 1
        try:
            for i in range(bg_min_i - num_bgwin, bg_min_i + num_bgwin):
                for j in range(bg_min_j - num_bgwin, bg_min_j + num_bgwin):
                    local_map = scale_img_LAB[np.max([(i - num_bgwin), 0]): np.min([(i + num_bgwin), (H - 1)]),
                                np.max([(j - num_bgwin), 0]): np.min([(j + num_bgwin), (W - 1)]), :]
                    r_h, r_w, _ = local_map.shape
                    local_map = local_map.reshape((r_h * r_w, 3))

                    var = np.mean(local_map.std(axis=0))
                    if var <= bg_min_var: 
                        bg_min_var = var
                        bg_min_texture_i = i
                        bg_min_texture_j = j
            if bg_min_texture_i >= H-1 or bg_min_texture_i<1 or bg_min_texture_j >= W-1 or bg_min_texture_j<1:
                bg_min_texture_i = bg_min_i
                bg_min_texture_j = bg_min_j
        except:
            print('something wrong bg_min_var!')
            bg_min_texture_i = bg_min_i
            bg_min_texture_j = bg_min_j

        bg_min_i = bg_min_texture_i
        bg_min_j = bg_min_texture_j

    #前点纠正
    if mode == 'random' or mode == 'bestbg':
        line = np.zeros(skeleton_bg_label.shape)
        line[fg_min_i, fg_min_j] = 255
        line[bg_min_i, bg_min_j] = 255
        for i in range(np.min([bg_min_i, fg_min_i]), np.max([bg_min_i, fg_min_i])):
            y = int(((bg_min_j - fg_min_j) / (bg_min_i - fg_min_i)) * (i - fg_min_i) + fg_min_j)
            line[i, y] = 255
            try:
                line[i, y - 1] = 255
                line[i, y + 1] = 255
            except:
                print('something wrong !')
                continue
        #####################有没有相交的点###########################
        crosspoint = line * skeleton_fg_label
        for i in range(np.max([(fg_min_i - num_parameter), 0]), np.min([(fg_min_i + num_parameter), (H - 1)])):  #
            for j in range(np.max([(fg_min_j - num_parameter), 0]), np.min([(fg_min_j + num_parameter), (H - 1)])):
                crosspoint[i, j] = 0
        # print('crosspoint.shape', crosspoint.shape)
        if np.sum(crosspoint) != 0: 
            for i in range(0, H):
                for j in range(0, W):
                    if crosspoint[i, j] != 0:
                        fg_min_i = i
                        fg_min_j = j
                        break

    return fg_min_i, fg_min_j, bg_min_i, bg_min_j

#生成纹理
def TextureGenerate(img, bg_min_texture_i, bg_min_texture_j, num_bgwin):
    #i_step = bg_min_texture_i - fg_min_i
    #j_step = bg_min_texture_j - fg_min_j
    H, W, _ = img.shape
    new_H, new_W, _ = img[
                      np.max([(bg_min_texture_i - num_bgwin), 0]): np.min([(bg_min_texture_i + num_bgwin), (H - 1)]),
                      np.max([(bg_min_texture_j - num_bgwin), 0]): np.min([(bg_min_texture_j + num_bgwin), (W - 1)]),
                      :].shape
    #i_num = np.abs(i_step) // new_H + 1
    #j_num = np.abs(j_step) // new_W + 1
    figure_img = np.zeros((H + 2 * new_H, W + 2 * new_W, 3))

    left_i = np.max([(bg_min_texture_i - num_bgwin), 0]) + new_H
    left_j = np.max([(bg_min_texture_j - num_bgwin), 0]) + new_W
    #right_i = np.min([(bg_min_texture_i + num_bgwin), (H - 1)]) + new_H
    #right_j = np.min([(bg_min_texture_j + num_bgwin), (W - 1)]) + new_W

    x00 = left_i // new_H + 1
    y00 = left_j // new_W + 1
    x11 = (H + 2 * new_H - left_i) // new_H
    y11 = (W + 2 * new_W - left_j) // new_W
    # print(range(-x00, x11, 1))
    # print(range(-y00, y11, 1))
    for i in range(-x00 + 1, x11, 1):
        for j in range(-y00 + 1, y11, 1):
            if np.abs(i) % 2 == 0 and np.abs(j) % 2 == 0:
                figure_img[left_i + int(new_H * i): left_i + new_H + int(new_H * i),
                left_j + int(new_W * j): left_j + new_W + int(new_W * j), :] = img[np.max(
                    [(bg_min_texture_i - num_bgwin), 0]): np.min([(bg_min_texture_i + num_bgwin), (H - 1)]), np.max(
                    [(bg_min_texture_j - num_bgwin), 0]): np.min([(bg_min_texture_j + num_bgwin), (W - 1)]), :]
            elif np.abs(i) % 2 == 1 and np.abs(j) % 2 == 0:

                figure_img[left_i + int(new_H * i): left_i + new_H + int(new_H * i),
                left_j + int(new_W * j): left_j + new_W + int(new_W * j), :] = np.flip(
                    img[np.max([(bg_min_texture_i - num_bgwin), 0]): np.min([(bg_min_texture_i + num_bgwin), (H - 1)]),
                    np.max([(bg_min_texture_j - num_bgwin), 0]): np.min([(bg_min_texture_j + num_bgwin), (W - 1)]), :],
                    axis=0)
            elif np.abs(i) % 2 == 0 and np.abs(j) % 2 == 1:

                figure_img[left_i + int(new_H * i): left_i + new_H + int(new_H * i),
                left_j + int(new_W * j): left_j + new_W + int(new_W * j), :] = np.flip(
                    img[np.max([(bg_min_texture_i - num_bgwin), 0]): np.min([(bg_min_texture_i + num_bgwin), (H - 1)]),
                    np.max([(bg_min_texture_j - num_bgwin), 0]): np.min([(bg_min_texture_j + num_bgwin), (W - 1)]), :],
                    axis=1)
            else:

                figure_img[left_i + int(new_H * i): left_i + new_H + int(new_H * i),
                left_j + int(new_W * j): left_j + new_W + int(new_W * j), :] = np.flip(np.flip(
                    img[np.max([(bg_min_texture_i - num_bgwin), 0]): np.min([(bg_min_texture_i + num_bgwin), (H - 1)]),
                    np.max([(bg_min_texture_j - num_bgwin), 0]): np.min([(bg_min_texture_j + num_bgwin), (W - 1)]), :],
                    axis=0), axis=1)

    final_figure_img = figure_img[new_H:new_H + H, new_W:new_W + W, :]
    return final_figure_img


def y_sin(x, flag):
    if flag == 0:
        return 1 + (x ** 2) / 2
    elif flag == 1:
        return 1 - math.sin(math.pi * x) * x * 2 / 3
    elif flag == 2:
        return 1 + math.sin(math.pi * x) * x * 2 / 3
    elif flag == 3:
        return 1 - (x ** 2) / 2
    else:
        return 1

def InsertLabelGenerate(bg_label, fg_min_i, fg_min_j, bg_min_i, bg_min_j,num_parameter): 
    H, W = bg_label.shape
    i_step = bg_min_i - fg_min_i
    j_step = bg_min_j - fg_min_j

    if int(np.min([np.abs(i_step), np.abs(j_step)]) / 4) == 0:
        num_cross_step = 0
    else:
        num_cross_step = np.random.randint(0, int(np.min([np.abs(i_step), np.abs(j_step)]) / 4))
    num_i_step = np.abs(i_step) - num_cross_step
    num_j_step = np.abs(j_step) - num_cross_step

    #insert_label的形状散装
    step_list = []
    for i in range(num_i_step):
        step_list.append(0)
    for i in range(num_j_step):
        step_list.append(1)
    for i in range(num_cross_step):
        step_list.append(2)
    random.shuffle(step_list)

    step_num = len(step_list)
    step_line = np.zeros(bg_label.shape)
    img_line = np.zeros(bg_label.shape)
    new_step_i = fg_min_i
    new_step_j = fg_min_j
    s_num = 0
    flag_left = random.randint(0, 5)
    flag_right = random.randint(0, 5)
    random_final_left = random.randint(6, 14) / 10
    random_final_right = random.randint(6, 14) / 10 
    for s in step_list:
        num_parameter_left = num_parameter * y_sin(s_num * random_final_left / step_num, flag_left)
        num_parameter_right = num_parameter * y_sin(s_num * random_final_right / step_num, flag_right)
        num_parameter_left = int(num_parameter_left)
        num_parameter_right = int(num_parameter_right)

        if s == 0:
            new_step_i = int(new_step_i + i_step / np.abs(i_step))
        elif s == 1:
            new_step_j = int(new_step_j + j_step / np.abs(j_step))
        else:
            new_step_i = int(new_step_i + i_step / np.abs(i_step))
            new_step_j = int(new_step_j + j_step / np.abs(j_step))
        step_line[new_step_i, new_step_j] = 255
        #print()
        if flag_left % 2 == 1:
            #print('111')
            img_line[np.max([(new_step_i - num_parameter_left), 0]): np.min([(new_step_i + num_parameter_right), (H - 1)]), np.max([(new_step_j - num_parameter_left), 0]): np.min([(new_step_j + num_parameter_right), (W - 1)])] = 1
        else:
            #print('222')
            img_line[
            np.max([(new_step_i - num_parameter_left), 0]): np.min([(new_step_i + num_parameter_left), (H - 1)]),
            np.max([(new_step_j - num_parameter_left), 0]): np.min([(new_step_j + num_parameter_left), (W - 1)])] = 1
        s_num = s_num + 1
    return img_line


#寻找最大的块
def remove_small_points(map):
    # img = cv2.imread(image, 0) 
    img_label, num = measure.label(map, neighbors=8, return_num=True)
    props = measure.regionprops(img_label)

    resMatrix = np.zeros(img_label.shape)
    max_area = 0
    for i in range(0, len(props)):
        if props[i].area > max_area:
            max_area = props[i].area
            final = i
            # print('i', i)
            # print('props[i].area', props[i].area)
    tmp = (img_label == final + 1).astype(np.uint8)

    resMatrix += tmp 
    resMatrix *= 255
    return resMatrix

