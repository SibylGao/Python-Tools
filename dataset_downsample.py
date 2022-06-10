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
# ##training set
# img_source_path = "E:\\CVPMVSNet\\dtu-train-128\\Rectified\\"
# img_output_path = "F:\\CVP_DATA_RESIZE\\train\\Rectified\\"
# pfm_source_path = "E:\\CVPMVSNet\\dtu-train-128\\Depths\\"
# pfm_output_path = "F:\\CVP_DATA_RESIZE\\train\\Depths\\"

#eval_set
img_source_path = "E:\\CVPMVSNet\\dtu-test-1200\\Rectified\\"
img_output_path = "F:\\CVP_DATA_RESIZE\\eval\\Rectified\\"



def RGB_train_img():
    for i in range(129):
        source_img_path = img_source_path + "scan{}_train\\".format(i)
        save_img_path = img_output_path + "scan{}_train\\".format(i)
        if os.path.exists(source_img_path):
            f_list = os.listdir(source_img_path)
            for j in f_list:
                # img_src = cv2.imread(source_img_path + j, 1)   #opencv读取的是BGR图像！！注意！！！
                # img_downsample = cv2.pyrDown(img_src,dstsize=(80, 64))  #用这个函数RGB值会产生变化
                # img_src = Image.open(source_img_path + j) 
                img_src = mpimg.imread(source_img_path + j)
                img_downsample = cv2.resize(img_src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                # img_downsample = cv2.resize(img_src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                img = save_img_path + j
                if not os.path.exists(save_img_path):
                    os.makedirs(save_img_path)
                imageio.imwrite(img , img_downsample)

def RGB_eval_img():
    for i in range(129):
        source_img_path = img_source_path + "scan{}\\".format(i)
        save_img_path = img_output_path + "scan{}\\".format(i)
        if os.path.exists(source_img_path):
            f_list = os.listdir(source_img_path)
            for j in f_list: 
                img_src = mpimg.imread(source_img_path + j)
                img_downsample = cv2.resize(img_src, (1200,1600), interpolation=cv2.INTER_LINEAR)
                # img_downsample = cv2.resize(img_src, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                img = save_img_path + j
                if not os.path.exists(save_img_path):
                    os.makedirs(save_img_path)
                imageio.imwrite(img , img_downsample)
           
def pfm_file():
    for i in range(129):
        source_pfm_path = pfm_source_path + "scan{}_train\\".format(i)
        save_pfm_path = pfm_output_path + "scan{}_train\\".format(i)
        if os.path.exists(source_pfm_path):
            f_list = os.listdir(source_pfm_path)
            if not os.path.exists(save_pfm_path):
                os.makedirs(save_pfm_path)
            for j in f_list:
                pfm_file = source_pfm_path + j
                pfm_save_path = save_pfm_path + j
                ##用读取图像的代码好像行不通
                # pfm = imageio.imread(pfm_file)
                # pfm = np.array(pfm)
                # pfm_downsample = cv2.resize(pfm, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                # imageio.imwrite(pfm_save_path , pfm_downsample)
                pfm, scale = readPFM(pfm_file)
                # 在这里下采样
                pfm_downsample = cv2.resize(pfm, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                # print(scale)
                writePFM(pfm_save_path, pfm_downsample)


##SenceFlow处理PFM文件的代码
def readPFM(file):
    file = open(file, 'rb')
 
    color = None
    width = None
    height = None
    scale = None
    endian = None
 
    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
 
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')
 
    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
 
    data = np.fromfile(file, endian + 'f')


    shape = (height, width, 3) if color else (height, width)
 
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
 
def writePFM(file, image, scale=1):
    file = open(file, 'wb')
 
    color = None
 
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
 
    image = np.flipud(image)
 
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
 
    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))
 
    endian = image.dtype.byteorder
 
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
 
    file.write('%f\n'.encode() % scale)
 
    image.tofile(file)


def main():
    # RGB_train_img()
    RGB_eval_img()
    # pfm_file()
   


if __name__ == '__main__':
    main()
