import cv2 as cv
import numpy as np 



def Sobel(srcImage, dstImage):
    #a = 1
    
    Wx = np.array([[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]])
    Wy = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])
    
    
    # cv.imshow("Gradient X", Gx)
    # cv.imshow("Gradient Y", Gy)
    if dstImage == None:
        return 0
    return 1
def Prewitt(srcImage, dstImage):
    #
    Wx = np.array([[1/3, 0, -1/3], [1/3, 0, -1/3], [1/3, 0, -1/3]])
    Wy = np.array([[-1/3, -1/3, -1/3], [0, 0, 0], [1/3, 1/3, 1/3]])


    # cv.imshow("Gradient X", Gx)
    # cv.imshow("Gradient Y", Gy)

    if dstImage == None:
        return 0
    return 1
def Laplapce(srcImage, dstImage):
    #
    if dstImage == None:
        return 0
    return 1
def Canny(srcImage, dstImage):
    #
    if dstImage == None:
        return 0
    return 1
def EdgeDetection(srcImage, dstImage):
    #
    return 1