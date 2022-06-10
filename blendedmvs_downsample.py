from cv2 import cv2
import os
import imageio
from PIL import Image
import matplotlib.image as mpimg
import re
import numpy as np
import uuid
from scipy import misc
import numpy as np
import sys
import matplotlib.pyplot as plt
from shutil import copyfile


root_path = "G:\\dataset_low_res\\"
all_list = "G:\\blendedmvs_cvpformat2\\all_list.txt"

output_path = "G:\\blendedmvs_cvpformat3\\"

def RGB_eval_img():
    dirs = os.listdir(root_path)
    with open(all_list) as f:
        lines_without_shift = f.readlines()
        lines = [line.rstrip() for line in lines_without_shift]
        # lines = f.readlines()
        # count = 1
    for line in lines:
        scan_root = root_path + line + "\\"
        # output_scan_root = output_path + "scan" + str(count) + "\\"
        # count = count + 1
        output_scan_root = output_path + line + "\\"
        if not os.path.exists(output_scan_root):
                    os.makedirs(output_scan_root)
        output_img_path = output_scan_root + "blended_images\\"
        if not os.path.exists(output_img_path):
                    os.makedirs(output_img_path)
        scan_img_path = scan_root + "blended_images\\"
        imgs = os.listdir(scan_img_path)
        for ig in imgs:
            source_img_path = scan_img_path + ig
            save_img_path = output_img_path + ig
            if not os.path.exists(source_img_path):
                continue
            img_src = mpimg.imread(source_img_path)
            img_downsample = cv2.resize(img_src, (160,128), interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(save_img_path , img_downsample)


def rescale_pfm():
    dirs = os.listdir(root_path)
    with open(all_list) as f:
        lines_without_shift = f.readlines()
        lines = [line.rstrip() for line in lines_without_shift]
        # lines = f.readlines()
        count = 1
    for line in lines:
        scan_root = root_path + line + "\\"
        output_scan_root = output_path + line + "\\"
        if not os.path.exists(output_scan_root):
                    os.makedirs(output_scan_root)
        output_img_path = output_scan_root + "rendered_depth_maps\\"
        if not os.path.exists(output_img_path):
                    os.makedirs(output_img_path)
        scan_img_path = scan_root + "rendered_depth_maps\\"
        imgs = os.listdir(scan_img_path)
        for ig in imgs:
            source_img_path = scan_img_path + ig
            save_img_path = output_img_path + ig
            if not os.path.exists(source_img_path):
                continue
            img_src = mpimg.imread(source_img_path)
            img_downsample = cv2.resize(img_src, (160,128), interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(save_img_path , img_downsample)

def Cam_eval():
    with open(all_list) as f:
        lines_without_shift = f.readlines()
        lines = [line.rstrip() for line in lines_without_shift]
        # lines = f.readlines()
    for line in lines:
        scan_root = root_path + line + "\\"
        output_scan_root = output_path + line + "\\"
        if not os.path.exists(output_scan_root):
                    os.makedirs(output_scan_root)
        output_cam_path = output_scan_root + "cams\\"
        if not os.path.exists(output_cam_path):
                    os.makedirs(output_cam_path)
        scan_cam_path = scan_root + "cams\\"
        cam_num = os.listdir(scan_cam_path)
        for num in cam_num:
            cam_source_file = scan_cam_path + num
            # mvs_cam_short_path = scan_cam_path  + num
            # with open(mvs_cam_short_path) as f:
            #     mvs_cam_short_lines = f.readlines()
            #     mvs_cams_range_line = mvs_cam_short_lines[11]
            if not os.path.exists(cam_source_file):
                continue
            if num=="pair.txt":
                continue
            with open(cam_source_file) as f:
                lines_without_shift = f.readlines()
                lines = [line.rstrip() for line in lines_without_shift] 
                intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
                interval = np.fromstring(lines[11], dtype=np.float32, sep=' ')
                depth_min = interval[0]
                depth_interval = interval[1]/2
                data_1 = intrinsics[0,:]/4.8
                data_2 = intrinsics[1,:]/4.5
                intrinsics_downsample = intrinsics
                intrinsics_downsample[0,:] = data_1
                intrinsics_downsample[1,:] = data_2
                line_1 = ' '.join([str(x) for x in intrinsics_downsample[0,:]]) + ' \n'
                line_2 = ' '.join([str(x) for x in intrinsics_downsample[1,:]]) + ' \n' 
                line_3 = str(depth_min) + ' ' + str(depth_interval) + ' \n'
                lines_without_shift[7] = line_1
                lines_without_shift[8] = line_2
                lines_without_shift[11] = line_3
                save_filename = output_cam_path + num
                save_file = open(save_filename,'w+')
                save_file.writelines(lines_without_shift)


def piar_txt():
    with open(all_list) as f:
        lines_without_shift = f.readlines()
        lines = [line.rstrip() for line in lines_without_shift]
        # lines = f.readlines()
    for line in lines:
        scan_root = root_path + line + "\\"
        output_scan_root = output_path + line + "\\"
        if not os.path.exists(output_scan_root):
                    os.makedirs(output_scan_root)
        output_cam_path = output_scan_root + "cams\\"
        if not os.path.exists(output_cam_path):
                    os.makedirs(output_cam_path)
        scan_pair_path = scan_root + "cams\\" + "pair.txt"
        output_cam_path = output_cam_path + "pair.txt"
        copyfile(scan_pair_path, output_cam_path)
       


def main():
    RGB_eval_img()
    Cam_eval()
    piar_txt()

if __name__ == '__main__':
    main()