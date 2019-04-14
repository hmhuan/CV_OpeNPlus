#include "HarrisDetector.h"

int main()
{
	Mat src, harris;
	src = imread("D:/leaf.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	harris = DetectHarris(src, 2, 3, 0.04);
	imshow("harris", harris);

	waitKey(0);

	return 0;
}
