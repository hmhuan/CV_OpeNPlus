import cv2
import numpy as np
import sys
from scipy import signal

# read command-line arguments
filename = sys.argv[1]

# load and display original image
img = cv2.imread(filename, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Wx = np.array([[1.0, 0, -1.0], [1.0, 0, -1.0], [1.0, 0, -1.0]])
Wy = np.array([[1.0, 1.0, 1.0], [0, 0, 0], [-1.0, -1.0, -1.0]])

(m, n) = gray.shape
Gx, Gy, Edge = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))

def cast_uint(a):
    if a > 255:
        return 255
    if a < 0:
        return 0
    return a

for i in range(1, m - 1):
    for j in range(1, n - 1):
        #Slice matrix
        temp = gray[i - 1 : i + 2, j - 1 : j + 2]
        #calc Gx(i, j), Gy(i, j)
        convX = signal.convolve2d(Wx, temp, mode = 'valid').reshape(1)
        convY = signal.convolve2d(Wy, temp, mode = 'valid').reshape(1)
        Gx[i, j] = cast_uint(convX)
        Gy[i, j] = cast_uint(convY)
        Edge[i, j] = cast_uint(convX + convY)
cv2.imshow("original", img)
cv2.imshow("Gray", gray)

cv2.imshow("Gx", Gx)
cv2.imshow("Gy", Gy)

cv2.waitKey(0)

cv2.imshow("Sobel", Edge)
cv2.waitKey(0)

# # Demo test numpy process for matrix image
# a = np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
# print(a)
# b = a[0 : 3, 0 : 3].reshape(3, 3)
# Wx = np.array([[-1/4, 0, 1/4], [-2/4, 0, 2/4], [-1/4, 0, 1/4]])
# conv = np.uint8(signal.convolve2d(Wx, b, mode = 'valid').reshape(1))

# print(b)
# print(Wx)
# print(conv)