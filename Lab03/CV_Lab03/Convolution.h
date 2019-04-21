#pragma once
#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/*
Kiểm tra xem x, y có hợp lệ
*/
bool isInRange(int x, int y, int height, int width)
{
	if (x < 0 || y < 0 || x >= height || y >= width)
		return false;
	return true;
}

/*
Tính tổng của ma trận với kích thước blocksize, tại vị trí x, y
*/
float sumOfMat(const Mat & mat, int blockSize, int x, int y)
{
	float sum = 0.0f;
	int height = mat.rows, width = mat.cols;

	int halfBlockSize = blockSize / 2;
	for (int i = -halfBlockSize; i <= halfBlockSize; i++)
		for (int j = -halfBlockSize; j <= halfBlockSize; j++)
			if (isInRange(x + i, y + j, height, width))
				sum += mat.at<float>(x + i, y + j);
	return sum;
}


/*
Phép convolve cho ảnh 1 channel
Input: srcImg - ảnh gốc
		kernel - mảng chứa các phần tử của ma trận để tính tích chập
Output: Convolution - ảnh tích chập kết quả
*/
Mat convolve(Mat & srcImg, vector<float> kernel, int ksize)
{
	Mat Convolution;
	int height = srcImg.rows, width = srcImg.cols;
	int n = ksize * ksize;
	int halfKsize = ksize / 2;
	vector<int> dx;
	vector<int> dy;
	// Khởi tạo ảnh kết quả convolution có kích thước bằng ảnh gốc
	Convolution.create(height, width, CV_32F);
	// Khởi tạo offset kích thước bằng kerner
	for (int i = halfKsize; i >= -halfKsize; i--)
		for (int j = halfKsize; j >= -halfKsize; j--)
		{
			dx.push_back(i);
			dy.push_back(j);
		}
	/*

	*/
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			float sum = 0.0;
			for (int k = 0; k < n; k++)
				if (isInRange(i + dx[k], j + dy[k], height, width))
				{
					sum += srcImg.at<uchar>(i + dx[k], j + dy[k]) * kernel[k];
				}
			Convolution.at<float>(i, j) = sum;//saturate_cast<uchar>(sum);
 		}
	return Convolution;
}