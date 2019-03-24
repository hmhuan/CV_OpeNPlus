import sys
import cv2 as cv
import numpy as np 
import EdgeDetector as ed

if __name__ == "__main__":

    #List argv
    fileName = sys.argv[1]
    code = int(sys.argv[2])
    result = 0
    print(code)
    image = cv.imread(fileName)
    dstImage = None

    
    cv.imshow("Source",image)
    
    if (code == 1):
        #doc them tham so
        result = ed.Sobel(image, dstImage)
    elif (code == 2):
        #doc them tham so
        result = ed.Prewitt(image, dstImage)
    elif (code == 3):
        #doc them tham so
        result = ed.Laplace(image, dstImage)
    elif (code == 4):
        result = ed.Canny(image, dstImage)

    if result != 0:
        cv.imshow("Result", dstImage)
    cv.waitKey(0)
    cv.destroyAllWindows()