#pragma once

#include "opencv2\opencv.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat DerivativesProduct(Mat& Ix, Mat& Iy)
{
	/// Init product image
	Mat product(Ix.rows, Ix.cols, Ix.type());

	float *pxData, *pyData, *pData;
	float *pxRow, *pyRow, *pRow;
	pxData = (float*)Ix.data;
	pyData = (float*)Iy.data;
	pData = (float*)product.data;

	//width là chiều rộng ảnh, height là chiều cao ảnh
	int width = Ix.cols, height = Ix.rows;
	//widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = Ix.step[0];

	int nChannels = Ix.channels();

	for (int i = 0; i < height; i++, pxData += widthStep, pyData += widthStep, pData += widthStep)
	{
		pRow = pData;
		pxRow = pxData;
		pyRow = pyData;
		for (int j = 0; j < width; j++, pxRow += nChannels, pyRow += nChannels, pRow += nChannels)
		{
			auto x = pxRow[0];
			auto y = pyRow[0];
			auto res = x * y;
			*pRow = res;
		}
	}

	return product;
}

/**
* Corner detector using Harris algorithm
* Arguments:
*	@srcImg: source image (grayscale)
*	@blockSize: the size of neighbourhood considered for corner detection
*	@ksize: aperture parameter of Sobel derivative used
*	@k: Harris detector free parameter
* Return: the respone of detector at each pixel - R value
**/
Mat DetectHarris(const Mat& srcImg, int blockSize, int ksize, float k)
{
	// Init @sobelX and @sobelY image
	Mat Ix, Iy;
	Mat Ix2, Iy2, Ixy;
	Mat Sx2, Sy2, Sxy;
	Mat R;

	// 1. Compute x and y derivatives of @srcImg
	Sobel(srcImg, Ix, CV_32F, 1, 0, ksize, 1, 0, BORDER_DEFAULT);
	Sobel(srcImg, Iy, CV_32F, 1, 0, ksize, 1, 0, BORDER_DEFAULT);

	// 2. Compute products of derivatives at every pixel
	Ix2 = DerivativesProduct(Ix, Ix);
	Iy2 = DerivativesProduct(Iy, Iy);
	Ixy = DerivativesProduct(Ix, Iy);

	// 3. Compute the sum of derivatives' product at each pixel
	Sobel(Ix2, Sx2, CV_32F, 1, 1, ksize, 1, 0, BORDER_DEFAULT);
	Sobel(Iy2, Sy2, CV_32F, 1, 1, ksize, 1, 0, BORDER_DEFAULT);
	Sobel(Ixy, Sxy, CV_32F, 1, 1, ksize, 1, 0, BORDER_DEFAULT);

	// 4. Compute the respone of the detector at each pixel
	float *pSx2Data, *pSy2Data, *pSxyData, *pRData;
	float *pSx2Row, *pSy2Row, *pSxyRow, *pRRow;

	pSx2Data = (float*)Sx2.data;
	pSy2Data = (float*)Sy2.data;
	pSxyData = (float*)Sxy.data;
	pRData = (float*)R.data;

	/// width là chiều rộng ảnh, height là chiều cao ảnh
	int width = srcImg.cols, height = srcImg.rows;
	/// nChannels là số kênh màu
	int nChannels = srcImg.channels();
	/// widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = srcImg.step[0];

	for (int i = 0; i < height; i++, pSx2Data += widthStep, pSy2Data += widthStep, pSxyData += widthStep, pRData += widthStep)
	{
		pSx2Row = pSx2Data;
		pSy2Row = pSx2Data;
		pSxyRow = pSxyData;
		pRRow = pRData;
		for (int j = 0; j < width; j++, pSx2Row++, pSy2Row++, pSxyRow++, pRRow++)
		{
			// 4'. Define at each pixel (i, j) matrix M
			Mat M = (Mat_<float>(2, 2) << pSx2Row[0], pSxyRow[0], pSxyRow[0], pSy2Row[0]);
			/// Compute R
			float traceM = trace(M)[0];
			pRRow[0] = determinant(M) - k * traceM * traceM;
		}
	}

	return R;
}
