#pragma once
#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat ImageWithFeature(const Mat&srcImg, const Mat& R, float Threshold)
{
	/**
	*	@srcImg: source image (color image)
	*	@R: the matrix with R value in pixel. 
	*	@Threshold
	**/
	Mat dstImg;
	if (srcImg.type() != CV_8UC3)
		cvtColor(srcImg, dstImg, CV_GRAY2BGR);
	else
		dstImg = srcImg.clone();
	//float *pRData = (float *)R.data, *pRRow;
	uchar *pData = (uchar *)dstImg.data, *pRow;

	// width là chiều rộng mat R, height là chiều cao mat R
	int width = dstImg.cols, height = dstImg.rows;
	// widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStepDst = dstImg.step[0];

	float valR;
	for (int i = 0; i < height; i++, pData += widthStepDst)
	{
		pRow = pData;
		for (int j = 0; j < width; j++, pRow += 3)
		{
			valR = R.at<float>(i, j);
			if (valR > Threshold)
			{
				// pRow[0] = 0; // Blue = 0
				// pRow[1] = 0; // Green = 0
				// pRow[2] = 255; // Red = 255
				circle(dstImg, Point(i, j), 3.0, Scalar(0, 0, 255), 2, 8);
			}
		}	
	}
	return dstImg;
}

bool isInRange(int x, int y, int height, int width)
{
	if (x < 0 || y < 0 || x >= height || y >= width)
		return false;
	return true;
}

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

Mat NonMaximumSuppression(Mat & R, int blockSize, float Threshold)
{
	Mat Nms;
	int halfBlockSize = blockSize / 2;
	int height = R.rows, width = R.cols;
	Nms.create(R.rows, R.cols, CV_8U);
	
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			float currVal = R.at<float>(i, j); // Xet gia tri tai i, j
			bool check = true;
			if (currVal > Threshold)
			{
				for (int x = -halfBlockSize; x <= halfBlockSize; x++)
				{
					for (int y = -halfBlockSize; y <= halfBlockSize; y++)
						if (isInRange(i + x, j + y, height, width))
							if (currVal < R.at<float>(i + x, j + y))
							{
								check = false;
								break;
							}
					if (!check)
						break;
				}
				if (check)
					Nms.at<uchar>(i, j) = 1;
				else
					Nms.at<uchar>(i, j) = 0;
			}
			else
				Nms.at<uchar>(i, j) = 0;
		}
	return Nms;
}

Mat DerivativesProduct(const Mat &Ix, const Mat& Iy)
{
	// Init product image
	Mat product(Ix.rows, Ix.cols, CV_32F);

	//width là chiều rộng ảnh, height là chiều cao ảnh
	int height = product.rows, width = product.cols;
	//widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = Ix.step[0];
	int nChannels = Ix.channels();

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float mul = Ix.at<uchar>(i, j) * Iy.at<uchar>(i, j) * 1.0f;
			product.at<float>(i, j) = mul;
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
Mat DetectHarris(const Mat& srcImg, int blockSize, int ksize, float k, float Threshold)
{
	// Init @sobelX and @sobelY image
	Mat blurImg;
	Mat Ix, Iy, sobelx, sobely;
	Mat Ix2, Iy2, Ixy;
	Mat R;
	
	// 1. Lọc ảnh với Gaussian để giảm nhiễu
	GaussianBlur(srcImg, blurImg, cv::Size(ksize, ksize), 0);
	
	// 2. Compute x and y derivatives of @srcImg sobel 3 x 3
	Sobel(blurImg, Ix, CV_8U, 1, 0, 3);
	Sobel(blurImg, Iy, CV_8U, 0, 1, 3);

	/*Convolution conv;
	vector<float> kernelX, kernelY;

	float Wx[9] = { 1.0/4, 0, -1.0/4, 2.0/4, 0, -2.0/4, 1.0/4, 0, -1.0/4};
	float Wy[9] = { 1.0/4, 2.0/4, 1.0/4, 0, 0, 0, -1.0/4, -2.0/4, -1.0/4};

	for (int i = 0; i < 9; i++)
	{
		kernelX.push_back(Wx[i]);
		kernelY.push_back(Wy[i]);
	}
	
	conv.SetKernel(kernelX, 3, 3);
	conv.DoConvolution(blurImg, Ix);

	conv.SetKernel(kernelY, 3, 3);
	conv.DoConvolution(blurImg, Iy);*/

	// 2. Compute products of derivatives at every pixel
	Ixy = DerivativesProduct(Ix, Iy);
	Ix2 = DerivativesProduct(Ix, Ix);
	Iy2 = DerivativesProduct(Iy, Iy);
	
	// 4. Compute the respone of the detector at each pixel
	R.create(srcImg.rows, srcImg.cols, CV_32F);
	float * pRData = (float*)R.data, *pRRow;

	// width là chiều rộng ảnh, height là chiều cao ảnh
	int width = srcImg.cols, height = srcImg.rows;
	// nChannels là số kênh màu
	int nChannels = srcImg.channels();
	// widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = R.step[0];

	float sumIx2, sumIy2, sumIxy, TraceM, detM;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			sumIx2 = sumOfMat(Ix2, blockSize, i, j);
			sumIy2 = sumOfMat(Iy2, blockSize, i, j);
			sumIxy = sumOfMat(Ixy, blockSize, i, j);
			detM = sumIx2 * sumIy2 - sumIxy * sumIxy;
			TraceM = sumIx2 + sumIy2;
			R.at<float>(i, j) = detM - k * TraceM * TraceM;
		}
	// 
	//Mat Nms;
	//int halfBlockSize = blockSize / 2;
	//Nms.create(height, width, CV_8UC1);
	//for (int i = 0; i < height; i++)
	//	for (int j = 0; j < width; j++)
	//	{
	//		float currVal = R.at<float>(i, j); // Xet gia tri tai i, j
	//		bool check = true;
	//		for (int x = -halfBlockSize; x <= halfBlockSize; x++)
	//		{
	//			for (int y = -halfBlockSize; y <= halfBlockSize; y++)
	//				if (isInRange(i + x, j + y, height, width))
	//				{
	//					float neighbor = R.at<float>(i + x, j + y);
	//					if (currVal < neighbor)
	//					{
	//						check = false;
	//						break;
	//					}
	//				}
	//			if (!check)
	//				break;
	//		}
	//		if (check)
	//			Nms.at<uchar>(i, j) = 1;
	//		else
	//			Nms.at<uchar>(i, j) = 0;
	//	}
	Mat Nms = NonMaximumSuppression(R, blockSize, Threshold);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			if (Nms.at<uchar>(i, j) == 0)
				R.at<float>(i, j) = 0;
		}
	return R;
}


Mat detectBlob(Mat image)
{
	/* Laplacian of Gaussian

	*/

	Mat dstImage;


	return dstImage;
}

Mat detectDOG(Mat image)
{
	/* Different of Gaussian

	*/
	Mat dstImage;
	return dstImage;
}
double matchBySIFT(Mat image1, Mat image2, int detector)
{
	/* SIFT

	*/
	return 0;
}
