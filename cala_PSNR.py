import math
import torch
import re
import torch.nn as nn
import numpy as np
from skimage.measure.simple_metrics import compare_psnr
import  os
import glob
import cv2
import argparse
from skimage import io as skio
from skimage import color as skco




def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype(np.float64) - min_val) / (max_val - min_val)
    return out


def calc_psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_metrics(img1, img2, crop_border, test_Y=True):
    #
    img1 = img1 / 255.
    img2 = img2 / 255.

    if test_Y and img1.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    height, width = img1.shape[:2]
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border, :]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:height-crop_border, crop_border:width-crop_border]
        cropped_im2 = im2_in[crop_border:height-crop_border, crop_border:width-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)
    return psnr, ssim

def calc_ssim(img1, img2):

    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="logs/PReNet6/", help='path to model and log files')
parser.add_argument("--rain_path", type=str, default="./datasets/test/Rain100L/PreNet_FBL/", help='path to training data')
parser.add_argument("--norain_path", type=str, default="./results/", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] ='0'
psnrs=[];ssims=[]
for num in range(1,101):

    rain_path=opt.rain_path+'rain-{:0>3d}'.format(num)
    norain_path=opt.norain_path+'norain-{:0>3d}'.format(num)
    x=cv2.imread(rain_path)
    x_true=cv2.imread(norain_path)
    ps,ss=calc_metrics(x,x_true,1)
    psnrs.append(ps)
    ssims.append(ss)
print('PSNR={:.4f},SSIM={:.4f}'.format(np.mean(psnrs),np.mean(ssims)))



