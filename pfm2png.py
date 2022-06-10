# -*- coding: UTF-8 -*-
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import cv2
 
def pfm_png_file_name(pfm_file_path,png_file_path):
    png_file_path={}
    for root,dirs,files in os.walk(pfm_file_path):
        for file in files:
            file = os.path.splitext(file)[0] + ".png"
            files = os.path.splitext(file)[0] + ".pfm"
            png_file_path = os.path.join(root,file)
            pfm_file_path = os.path.join(root,files)
            pfm_png(pfm_file_path,png_file_path)
 
def pfm_png(pfm_file_path,png_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channel = 3 if header == 'PF' else 1
        # 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match() 就返回 none
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")
 
        scale = float(pfm_file.readline().decode().strip())
        if scale < 0:
            endian = '<'    #little endlian
            scale = -scale
        else:
            endian = '>'    #big endlian
 
        disparity = np.fromfile(pfm_file, endian + 'f')
 
        img = np.reshape(disparity, newshape=(height, width))
        img = np.flipud(img)
        print(img)

        h,w = img.shape[0],img.shape[1]
        img_resize = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
        # 归一化
        # img = (img - np.min(img))/(np.max(img) - np.min(img))

        # gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        plt.imsave(os.path.join(png_file_path + ".png"), img)
        # plt.imsave(os.path.join(png_file_path + "_rescale"+ ".png"), img_resize)
 
def main():
    pfm_path = "G:\\tmp\\pfm\\"
    png_path = "G:\\tmp\\png\\"
    # pfm_file_dir = '/home/cc/下载/PSMNet-master/dataset/data_scene_flow/training/disp_occ_0'
    # png_file_dir = '/home/cc/下载/PSMNet-master/dataset/data_scene_flow/training/disp_occ_1'
    dirs = os.listdir(pfm_path)
    for scan in dirs:
        pfm_path_1 = pfm_path + scan + "\\"
        png_path_1 = png_path + scan + "\\"
        if not os.path.exists(png_path_1):
            os.makedirs(png_path_1)
        pfms = os.listdir(pfm_path_1)
        count = 0
        for pfm in pfms:
            pfm_path_2 = pfm_path_1 + pfm
            # png_path_2 = png_path_1 + pfm + ".png"
            png_path_2 = png_path_1 + pfm 
            count = count + 1
            pfm_png(pfm_path_2, png_path_2)
    # for k in range(128):
    #     pfm_path_1 = pfm_path + "scan{}_train\\".format(k)
    #     if os.path.exists(pfm_path_1):
    #         for g in range(48):
    #             pfm_path_2 = pfm_path_1 + "depth_map_{:0>4}.pfm".format(g)
    #             png_path_1 = png_path + "scan{}_train\\".format(k) 
    #             if not os.path.exists(png_path_1):
    #                 os.makedirs(png_path_1)
    #             png_path_2 = png_path_1 + "depth_map_{:0>4}.png".format(g)
    #             pfm_png(pfm_path_2, png_path_2)
   
if __name__ == '__main__':
    main()