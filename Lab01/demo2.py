import sys
import cv2 as cv

# ddepth = cv.CV_16S
# kernel_size = 3

src_gray = cv.imread("Lena.png", cv.IMREAD_GRAYSCALE)
dst = cv.Laplacian(src_gray, cv.CV_8U, 3)
cv.imshow("opencv", dst)
cv.waitKey(0)