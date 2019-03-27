import sys
import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt
from EdgeDetector import Sobel, Prewitt, Laplace, Canny

def show(name, srcImage, dstImage, Gx = None, Gy = None):

    if (name == "Sobel" or name == "Prewitt"):
        #Show GrayImage
        plt.subplot(2,2,1),plt.imshow(srcImage,cmap = 'gray')
        plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
        #Show final Image
        plt.subplot(2,2,2),plt.imshow(dstImage,cmap = 'gray')
        plt.title(name), plt.xticks([]), plt.yticks([])
        #Show Gx
        plt.subplot(2,2,3),plt.imshow(Gx,cmap = 'gray')
        plt.title(name + " X"), plt.xticks([]), plt.yticks([])
        #Show Gy
        plt.subplot(2,2,4),plt.imshow(Gy,cmap = 'gray')
        plt.title(name + "Y"), plt.xticks([]), plt.yticks([])
    else:
        #Show GrayImage
        plt.subplot(1,2,1),plt.imshow(srcImage,cmap = 'gray')
        plt.title('Grayscale'), plt.xticks([]), plt.yticks([])
        #Show final Image
        plt.subplot(1,2,2),plt.imshow(dstImage,cmap = 'gray')
        plt.title(name), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    operator = sys.argv[2]
    srcImage = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    if operator == "--sobel":
        Gx, Gy, dstImage = Sobel(srcImage)
        show( "Sobel", srcImage, dstImage, Gx, Gy)
    elif operator == "--prewitt":
        Gx, Gy, dstImage = Sobel(srcImage)
        show("Prewitt", srcImage, dstImage, Gx, Gy)
    elif operator == "--laplace":
        dstImage = Laplace(srcImage)
        show("Laplace", srcImage, dstImage)
    elif operator == "--canny":
        dstImage = Canny(srcImage)
        show("Canny", srcImage, dstImage)
