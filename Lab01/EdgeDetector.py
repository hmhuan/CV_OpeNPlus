import cv2 as cv
import numpy as np 

def saturate_cast(val):
    """ Fix value of a pixel to saturate with image type """
    if val < 0:
        val = 0
    if val > 255:
        val = 255
    return val


def Sobel(srcImage):
    """ Edge detector using Sobel operator
        Argument:
        @srcImage: source image
        Return: Destination image after applying Sobel operator
    """
    # Get height and width of @srcImage
    height, width = srcImage.shape
    # Create @dstImage with shape and type of @srcImage
    Gx = np.zeros((height, width), srcImage.dtype)
    Gy = np.zeros((height, width), srcImage.dtype)
    dstImage = np.zeros((height, width), srcImage.dtype)
    # Kernel of the derivative in x direction    
    Wx = np.array([[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]])
    # Kernel of the derivative in y direction    
    Wy = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])
    for row in range(1, height-1):
        for col in range(1, width-1):
            # The derivative in x direction of pixel[rol, col]
            fx = (srcImage[row-1:row+2, col-1:col+2] * Wx).sum()
            Gx[row, col] = saturate_cast(fx)
            # The derivative in y direction of pixel[rol, col]
            fy = (srcImage[row-1:row+2, col-1:col+2] * Wy).sum()
            Gy[row, col] = saturate_cast(fy)
            # Approximation of the gradient in that pixel
            dstImage[row, col] = saturate_cast(fx + fy)
    return Gx, Gy, dstImage

    
def Prewitt(srcImage, dstImage):
    """ Edge detector using Sobel operator
        Argument:
        @srcImage: source image
        Return: Destination image after applying Prewitt operator
    """
    # Get height and width of @srcImage
    height, width = srcImage.shape
    # Create @dstImage with shape and type of @srcImage
    Gx, Gy, dstImage = np.zeros((height, width), srcImage.dtype)
    # Kernel of the derivative in x direction  
    Wx = np.array([[1/3, 0, -1/3], [1/3, 0, -1/3], [1/3, 0, -1/3]])
    # Kernel of the derivative in y direction  
    Wy = np.array([[-1/3, -1/3, -1/3], [0, 0, 0], [1/3, 1/3, 1/3]])
    for row in range(1, height-1):
        for col in range(1, width-1):
            # The derivative in x direction of pixel[rol, col]
            fx = (srcImage[row-1:row+2, col-1:col+2] * Wx).sum()
            Gx[row, col] = saturate_cast(fx)
            # The derivative in y direction of pixel[rol, col]
            fy = (srcImage[row-1:row+2, col-1:col+2] * Wy).sum()
            Gy[row, col] = saturate_cast(fy)
            # Approximation of the gradient in that pixel
            dstImage[row, col] = saturate_cast(fx + fy)
    return Gx, Gy, dstImage


def Laplace(srcImage):
    """ Edge detector using Laplace operator
        Argument:
        @srcImage: source image
        Return: Destination image after applying Laplace operator
    """
    # Get height and width of @srcImage
    height, width = srcImage.shape
    # Create @dstImage with shape and type of @srcImage
    dstImage = np.zeros((height, width), srcImage.dtype)
    # Lapalce kernel
    WLap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    for row in range(1, height-1):
        for col in range(1, width-1):
            val = (srcImage[row-1:row+2, col-1:col+2] * WLap).sum()
            dstImage[row, col] = saturate_cast(val)
    return dstImage
