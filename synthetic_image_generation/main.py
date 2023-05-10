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
from function import *


path = '/data/DUTS/image'#image_path
name_list = os.listdir(path)
count = 20
no_list = []

for w in range(10):
    count = count + 1
    cnt = 0
    for name in name_list:

        try:
            print(name)
            cnt = cnt + 1
            print('count', count)
            print(cnt)
            #num_parameter = random.randint(5,15)
            #num_parameter = random.randint(15, 20)
            num_parameter = random.randint(10, 15)
            num_bgwin = 15

            #name = 'n03790512_2777.jpg'
            img = cv2.imread(path + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            mask = cv2.imread(
                '/data/DUTS/scribble/' + name.replace('.jpg', '.png'), #scribble_path
                cv2.IMREAD_GRAYSCALE)

            scale_img = img / 255
            scale_img_LAB = img_LAB / 255

            bg_label = np.copy(mask)
            fg_label = np.copy(mask)
            bg_label[bg_label == 1.] = 0
            bg_label[bg_label == 2.] = 1
            fg_label[fg_label == 2.] = 0
            fg_label[fg_label == 1.] = 1



            ######################################
            mode_flag = random.randint(0,10)#mode
            if mode_flag <= 4:
                mode = 'bestbg'
            elif mode_flag == 5:
                mode = 'near'
            else:
                mode = 'random'
            fg_min_i, fg_min_j, bg_min_i, bg_min_j = FindForeBackPoint(fg_label=fg_label, bg_label=bg_label, num_bgwin=num_bgwin, num_parameter=num_parameter, scale_img_LAB=scale_img_LAB, mode=mode)
            final_figure_img = TextureGenerate(img=img, bg_min_texture_i=bg_min_i, bg_min_texture_j=bg_min_j, num_bgwin=num_bgwin)
            img_line = InsertLabelGenerate(bg_label, fg_min_i, fg_min_j, bg_min_i, bg_min_j,num_parameter)

            # plt.figure('3')
            # plt.imshow(img_line)
            # plt.show()

            ####################################################
            reverse_fg_label = 1 - fg_label
            reverse_bg_label = 1 - bg_label

            intert_iou = fg_label * img_line#

            final_img_line = reverse_fg_label * img_line
            final_img_line = reverse_bg_label * final_img_line




            final_img_line = remove_small_points(final_img_line)

            final_img_line_final = final_img_line
            final_img_line_good = final_img_line/255
            rd = random.randint(0, 16)
            if rd == 0:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (3, 3), 2.5)
            if rd == 1:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (3, 3), 3.5)
            elif rd == 2:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (5, 5), 0.5)
            elif rd == 3:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (5, 5), 2.5)
            elif rd == 4:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (5, 5), 3.5)
            elif rd == 5:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (5, 5), 4.5)
            elif rd == 6:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (7, 7), 3.5)
            elif rd == 7:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (7, 7), 0.5)
            elif rd == 8:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (7, 7), 2.5)
            elif rd == 9:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (9, 9), 3.5)
            elif rd == 10:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (9, 9), 2.5)
            elif rd == 11:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (9, 9), 4.5)
            elif rd == 12:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (9, 9), 5.5)
            elif rd == 13:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (9, 9), 6.5)
            elif rd == 14:
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (9, 9), 1.5)
            elif rd == 15:
                # final_img_line_blur = final_img_line
                final_img_line_good = cv2.GaussianBlur(final_img_line_good, (7, 7), 1.5)
            else:
                final_img_line_good = final_img_line_good
            #############################################################################################################################################


            H, W, _ = img.shape
            shadow_win = np.zeros((50, 50, 3))
            shadow_img = np.zeros((H, W, 3))
            if random.randint(0, 4) == 10:    #shadow_win
                if random.randint(0, 1) == 0:
                    for i in range(50):
                        j = int(25 + 50 * math.sin(2 * math.pi * i / 50) / 4)
                        shadow_win[i, j, :] = 1
                else:
                    for j in range(50):
                        i = int(25 + 50 * math.sin(2 * math.pi * j / 50) / 4)
                        shadow_win[i, j, :] = 1
                # if random.randint(0, 2) == 0:

                for i in range(26, H - 26, 50):
                    for j in range(26, W - 26, 50):
                        shadow_img[i - 25:i + 25, j - 25:j + 25, :] = shadow_win
                if random.randint(0, 1) == 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    shadow_img = cv2.dilate(shadow_img, kernel, iterations=5)
                    shadow_img = cv2.GaussianBlur(shadow_img, (49, 49), 10)
                else:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    shadow_img = cv2.dilate(shadow_img, kernel, iterations=5)
                shadow_img = 1 - shadow_img / (random.randint(125, 500) / 100)
            else:
                shadow_img = np.ones((H, W, 3))

            #print('shadow_img.shape', shadow_img.shape)
            shadow_img = np.mean(shadow_img, axis=2)

            # plt.figure('3')
            # plt.imshow(final_img_line_good)
            # plt.show()
            ################################################################################################################################################
            aug_img = np.zeros(img.shape)
            aug_img[:, :, 0] = final_figure_img[:, :, 0] * final_img_line_good * shadow_img + img[:, :, 0] * (
                        1 - final_img_line_good)
            aug_img[:, :, 1] = final_figure_img[:, :, 1] * final_img_line_good * shadow_img + img[:, :, 1] * (
                        1 - final_img_line_good)
            aug_img[:, :, 2] = final_figure_img[:, :, 2] * final_img_line_good * shadow_img + img[:, :, 2] * (
                        1 - final_img_line_good)

            final_img_line_good = final_img_line_good * 255
            final_img_line_good = final_img_line_good.astype('int')

            # plt.figure('3')
            # plt.imshow(final_img_line_good)
            # plt.show()

            if not os.path.exists('synthetic/' + str(count) + '_img'):
                os.makedirs('synthetic/' + str(count) + '_img')
            if not os.path.exists('synthetic/' + str(count) + '_label'):
                os.makedirs('synthetic/' + str(count) + '_label')
            imageio.imsave('synthetic/' + str(count) + '_img/' + name, aug_img)
            imageio.imsave('synthetic/' + str(count) + '_label/' + name.replace('.jpg', '.png'), final_img_line)
            #print('ssssssssssssssssssssssssss')
        except:
            print('ssssssssssssssssssssssssss')
            no_list.append(name)
print(no_list)