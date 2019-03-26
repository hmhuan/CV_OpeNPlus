import cv2 as cv
import numpy as np

def convolve(srcImg, dstImg, kernel, isLaplace=True):
    height, width = srcImg.shape
    dstImg = np.zeros((height, width), srcImg.dtype)
    # dstImg[0] = srcImg[0]
    # dstImg[height-1] = srcImg[height-1]
    # dstImg[:height, 0] = srcImg[:height, 0]
    # dstImg[:height, width-1] = srcImg[:height, width-1]
    if isLaplace:
        for row in range(1, height-1):
            for col in range(1, width-1):
                sum = (srcImg[row-1:row+2, col-1:col+2] * kernel).sum()
                if sum > 255:
                    sum = 255
                if sum < 0:
                    sum = 0
                dstImg[row, col] = sum
