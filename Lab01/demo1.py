import cv2 as cv
import numpy as np
from convolution import convolve


src = cv.imread("Lena.png", cv.IMREAD_GRAYSCALE)
print(src.dtype)
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
Wx = np.array([[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]])
Wy = np.array([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]])
dstImg = None
convolve(src, dstImg, kernel)

# cv.imshow("Source", src)
cv.imshow("Destination", dstImg)
cv.waitKey(0)
cv.destroyAllWindows()

# image = cv.imread("D:\dreams.jpg", cv.IMREAD_GRAYSCALE)
# # # image = np.ones((5, 7), np.uint8)
# # image = np.array([[35,22,73,44,5,6,7], [122,120,84,8,9,12,9], [21,12,53,4,5,6,7], [2,0,4,8,9,12,9], [1,2,3,4,5,6,7]])
# # height, width= image.shape
# # print(height, width)

# # square = np.zeros((height, width), np.uint8)
# # square[0] = image[0]
# # square[height-1] = image[height-1]
# # square[:height, 0] = image[:height, 0]
# # square[:height, width-1] = image[:height, width-1]

# Wx = np.array([[0.25, 0, -0.25], [0.5, 0, -0.5], [0.25, 0, -0.25]])
# i, j = 122, 25
# point = image[i-1:i+2, j-1:j+2]
# x = (point * Wx).sum()
# if x > 255:
#     x = 255
# if x < 0:
#     x = 0
# print(x)
# # x = np.convolve(point[0], Wx[0], 'valid') + np.convolve(point[1], Wx[1], 'valid') + np.convolve(point[2], Wx[2], 'valid')
# # print(x)
# # print(point*Wx)

# # for row in range(1, height-1):
# #     for col in range(1, width-1):
# #         square[row, col] = image[i-1:i+2, j-1:j+2] * Wx).sum()





# # # cv.imshow("Numpy", square)
# # # cv.waitKey(0)
# # # cv.destroyAllWindows()

# # print(square)