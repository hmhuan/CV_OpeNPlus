import sys
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
from edge_detector import Sobel, Prewitt, Laplace

def show(srcImage, dstImage, Gx, Gy, name):
    plt.subplot(2,2,1),plt.imshow(srcImage,cmap = 'gray')
    plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(dstImage,cmap = 'gray')
    plt.title(name), plt.xticks([]), plt.yticks([])
    if Gx != None:
        plt.subplot(2,2,3),plt.imshow(Gx,cmap = 'gray')
        plt.title(name + " X"), plt.xticks([]), plt.yticks([])
    if Gy != None:
        plt.subplot(2,2,4),plt.imshow(Gy,cmap = 'gray')
        plt.title(name + "Y"), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    operator = sys.argv[2]
    srcImage = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    if operator == "--sobel":
        Gx, Gy, dstImage = Sobel(srcImage)
        show(srcImage, dstImage, Gx, Gy, "Sobel")
    elif operator == "--prewitt":
        Gx, Gy, dstImage = Sobel(srcImage)
        show(srcImage, dstImage, Gx, Gy, "Prewitt")
    elif operator == "--laplace":
        dstImage = Laplace(srcImage)
        show(srcImage, dstImage, Gx=None, Gy=None, name="Laplace")
