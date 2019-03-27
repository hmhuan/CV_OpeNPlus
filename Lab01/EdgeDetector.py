import cv2 as cv
import numpy as np 

def saturate_cast(val):
    """ Fix value of a pixel to saturate with image type """
    if int(val) < 0:
        return 0
    if int(val) > 255:
        return 255
    return int(val)


def Sobel(srcImage):
    """ Edge detector using Sobel operator
        Argument:
        @srcImage: source image
        Return: Destination image after applying Sobel operator
    """
    # Get height and width of @srcImage
    height, width = srcImage.shape
    # Create @dstImage with shape and type of @srcImage
    Gx = np.zeros((height, width))
    Gy = np.zeros((height, width))
    dstImage = np.zeros((height, width))
    # Kernel of the derivative in x direction    
    #Wx = np.array([[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]])
    Wx = np.array([[0.5], [1],[0.5]])
    Wy = np.array([[-0.5], [0], [0.5]])
    # Kernel of the derivative in y direction    
    #Wy = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])
    for row in range(1, height-1):
        for col in range(1, width-1):
            # The derivative in x direction of pixel[rol, col]
            temp = srcImage[row - 1 : row + 2, col - 1 : col + 2]
            fx = np.dot(Wx.T, np.dot(temp,Wy)).reshape(1)
            Gx[row, col] = saturate_cast(fx)
            # The derivative in y direction of pixel[rol, col]
            fy = np.dot(Wy.T, np.dot(temp, Wx)).reshape(1)
            Gy[row, col] = saturate_cast(fy)
            # Approximation of the gradient in that pixel
            dstImage[row, col] = saturate_cast(abs(fx + fy))
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
            temp = srcImage[row - 1 : row + 2, col - 1 : col + 2]
            # The derivative in x direction of pixel[rol, col]
            fx = (temp * Wx).sum()
            Gx[row, col] = saturate_cast(fx)
            # The derivative in y direction of pixel[rol, col]
            fy = (temp * Wy).sum()
            Gy[row, col] = saturate_cast(fy)
            # Approximation of the gradient in that pixel
            dstImage[row, col] = saturate_cast(abs(fx + fy))
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


def Hysteresis(gradNms, dstImage, x, y, low):
    
    if (x < 0 or y < 0 or x >= dstImage.shape[0] or y >= dstImage.shape[1] or dstImage[x, y] != 0):
        return 
    if (gradNms[x, y] < low):
        return
    dstImage[x, y] = 255
    
    Hysteresis(gradNms, dstImage, x - 1, y - 1, low)
    Hysteresis(gradNms, dstImage, x - 1, y, low)
    Hysteresis(gradNms, dstImage, x - 1, y + 1, low)
    Hysteresis(gradNms, dstImage, x , y - 1, low)
    Hysteresis(gradNms, dstImage, x , y + 1, low)
    Hysteresis(gradNms, dstImage, x + 1, y - 1, low)
    Hysteresis(gradNms, dstImage, x + 1, y, low)
    Hysteresis(gradNms, dstImage, x + 1, y + 1, low)

def Canny(srcImage):

    (m, n) = srcImage.shape
    lowThreshold = 30
    highThreshold = lowThreshold * 3
    # Initial Gradient X, Gradient Y and Edge
    Gx = np.zeros((m, n), dtype = srcImage.dtype)
    Gy = np.zeros((m, n), dtype = srcImage.dtype)
    G = np.zeros((m, n), dtype = srcImage.dtype)
    dstImage = np.zeros((m, n), dtype = srcImage.dtype)

    # Step 1: Gaussian filter
    blurImg = cv.GaussianBlur(srcImage, (5, 5), 0)

    # Step 2: Find Orientation at each pixel and Non-maximum suppression
    # Mat Wx, Wy
    Wx = np.array([[1], [2],[1]])
    Wy = np.array([[-1], [0], [1]])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            #Slice matrix
            temp = srcImage[i - 1 : i + 2, j - 1 : j + 2]
            #calc Gx(i, j), Gy(i, j)
            fx = np.dot(np.dot(Wy.T, temp), Wx).reshape(1) #(Wx * temp).sum()
            Gx[i, j] = saturate_cast(fx)
            fy = np.dot(np.dot(Wx.T, temp), Wy).reshape(1) #(Wy * temp).sum()
            Gy[i, j] = saturate_cast(fy)
            G[i, j] = saturate_cast(abs(fx + fy))

    gradNms = G.copy()
    Theta = np.fmod(np.arctan2(Gy, Gx) + np.pi, np.pi) / np.pi * 4
    #Non-maximum Supression
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            Ga, Gb = 0, 0
            if (Theta[i, j] == 0):
                Ga, Gb = G[i, j -1], G[i, j + 1]
            elif (Theta[i, j] == 1):
                Ga, Gb = G[i - 1, j + 1], G[i + 1, j - 1]
            elif (Theta[i, j] == 2):
                Ga, Gb = G[i - 1, j], G[i + 1, j]
            else:
                Ga, Gb = G[i - 1, j - 1], G[i + 1, j + 1]
            if (G[i, j] < Ga) or (G[i, j] < Gb):
                gradNms[i, j] = 0

    # Step 3: Hysteresis Thresholding  [L, H] Recursive
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if (gradNms[i, j] >= highThreshold):
                Hysteresis(gradNms, dstImage, i, j, lowThreshold)
    
    return dstImage
