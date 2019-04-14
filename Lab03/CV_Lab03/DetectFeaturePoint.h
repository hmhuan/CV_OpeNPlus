#pragma once
#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

//Mat detectHarris(Mat image, int Threshold)
//{
//	/*
//	GrayImage: anh xam
//	blurImage: anh lam tron bang Gaussian
//	Ix, Iy: Dao ham cua anh
//	*/
//	Mat dstImage, GrayImage, blurImage, Ix, Iy, G;
//	vector<int> offsets;
//	const float BYTE_TO_FLOAT = 1.0f;
//
//	//width là chiều rộng ảnh, height là chiều cao ảnh
//	int width = dstImage.cols, height = dstImage.rows;
//	//nChannels là số kênh màu
//	int nChannels = dstImage.channels();
//	//widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
//	int widthStep = dstImage.step[0];
//
//	// Khởi tạo offsets
//	int kHalfWidth = 3 >> 1, kHalfHeight = 3 >> 1, n = 9;
//	for (int y = -kHalfHeight; y <= kHalfHeight; y++)
//		for (int x = -kHalfWidth; x <= kHalfWidth; x++)
//			offsets.push_back(y * widthStep + x);
//
//	if (image.type() == CV_8UC1)
//		GrayImage = image.clone();
//	else
//		cvtColor(image, GrayImage, CV_BGR2GRAY);
//	GaussianBlur(GrayImage, blurImage, Size(5, 5), 1.4);
//
//	// 2. Tinh dao ham va huong cua anh
//	uchar *pBlur = (uchar *)blurImage.data, *pBlurRow;
//	uchar *pData = (uchar *)dstImage.data, *pRow;
//
//	float sobel = 1.0f;
//	float Wx[9] = { -1.0 / sobel, 0.0 / sobel, 1.0 / sobel,
//		-2.0 / sobel, 0.0 / sobel, 2.0 / sobel,
//		-1.0 / sobel, 0.0 / sobel, 1.0 / sobel };
//	float Wy[9] = { 1.0 / sobel, 2.0 / sobel, 1.0 / sobel,
//		0.0 / sobel, 0.0 / sobel, 0.0 / sobel,
//		-1.0 / sobel, -2.0 / sobel, -1.0 / sobel };
//	Ix.create(height, width, CV_32FC1);
//	Iy.create(height, width, CV_32FC1);
//	G.create(height, width, CV_32FC1);
//	// Tính đạo hàm theo hướng x, y và độ lớn gradiant ảnh
//	for (int i = 0; i < height; i++, pBlur += widthStep)
//	{
//		pBlurRow = pBlur;
//		for (int j = 0; j < width; j++, pBlurRow += nChannels)
//		{
//			float sumX = 0.0f, sumY = 0.0f, sum = 0.0f;
//			for (int k = 0; k < n; k++)
//			{
//				sumX += pBlurRow[offsets[k]] * Wx[n - 1 - k];
//				sumY += pBlurRow[offsets[k]] * Wy[n - 1 - k];
//			}
//			sum = hypot(sumX, sumY);
//			Ix.at<float>(i, j) = sumX;
//			Iy.at<float>(i, j) = sumY;
//			G.at<float>(i, j) = sum;
//		}
//	}
//	// 3. Xay dung ma tran M cho moi cua so tai moi diem anh va Tinh R
//	Mat R;
//	R.create(height, width, CV_32FC1);
//
//	float *pR = (float *)R.data, *pRrow;
//	float alpha = 0.05; // alpha in (0.04; 0.06)
//	for (int i = 0; i < height; i++, pR += widthStep)
//
//
//		// 4. Threshold R > T -> corner
//
//
//		return dstImage;
//}

Mat DerivativesProduct(const Mat &Ix, const Mat& Iy)
{
	/// Init product image
	Mat product(Ix.rows, Ix.cols, Ix.type());

	//float *pxData, *pyData, *pData;
	//float *pxRow, *pyRow, *pRow;
	//pxData = (float*)Ix.data;
	//pyData = (float*)Iy.data;
	//pData = (float*)product.data;

	////width là chiều rộng ảnh, height là chiều cao ảnh
	//int width = Ix.cols, height = Ix.rows;
	////widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	//int widthStep = Ix.step[0];

	//int nChannels = Ix.channels();

	//for (int i = 0; i < height; i++, pxData += widthStep, pyData += widthStep, pData += widthStep)
	//{
	//	pRow = pData;
	//	pxRow = pxData;
	//	pyRow = pyData;
	//	for (int j = 0; j < width; j++, pxRow += nChannels, pyRow += nChannels, pRow += nChannels)
	//	{
	//		pRow[0] = pxRow[0] * pyRow[0];
	//	}
	//}
	int height = product.rows, width = product.cols;
	for (int i = 0; i < height; i++) 
	{
		for (int j = 0; j < width; j++)
		{
			float mul = Ix.at<float>(i, j) * Iy.at<float>(i, j);
			product.at<float>(i, j) = mul;
		}
	}

	return product;
}

Mat ImageWithFeature(const Mat&srcImg, const Mat& R, float Threshold)
{
	/**
	*	@srcImg: source image (color image)
	*	@R: the matrix with R value in pixel. 
	*	@Threshold
	**/
	Mat dstImg = srcImg.clone();
	float *pRData = (float *)R.data, *pRRow;
	uchar *pData = (uchar *)dstImg.data, *pRow;
	// width là chiều rộng mat R, height là chiều cao mat R
	int width = R.cols, height = R.rows;
	// widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = R.step[0], widthStepDst = dstImg.step[0];

	float valR;
	for (int i = 0; i < height; i++, pData += widthStepDst)
	{
		pRow = pData;
		for (int j = 0; j < width; j++, pRow += 3)
		{
			valR = R.at<float>(i, j);
			if (valR > Threshold)
			{
				cout << valR << endl;
				pRow[0] = 0; // Blue = 0
				pRow[1] = 0; // Green = 0
				pRow[2] = 255; // Red = 255
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
	int *dx = new int[blockSize];
	int *dy = new int[blockSize];
	int height = mat.rows, width = mat.cols;
	for (int i = -blockSize / 2; i <= blockSize / 2; i++)
	{
		dx[i] = dy[i] = i;
	}
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			if (isInRange(x + dx[i], y + dy[j], height, width))
				sum += mat.at<float>(x + dx[i], y + dy[j]);
	return sum;
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
	Mat blurImg;
	Mat Ix, Iy;
	Mat Ix2, Iy2, Ixy;
	Mat Sx2, Sy2, Sxy;
	Mat R;
	
	GaussianBlur(srcImg, blurImg, cv::Size(ksize, ksize), 1.4);
	// 1. Compute x and y derivatives of @srcImg
	Sobel(blurImg, Ix, CV_32FC1, 1, 0, ksize, 1, 0, BORDER_DEFAULT);
	Sobel(blurImg, Iy, CV_32FC1, 0, 1, ksize, 1, 0, BORDER_DEFAULT);
	// 2. Compute products of derivatives at every pixel
	Ixy = DerivativesProduct(Ix, Iy);
	Ix2 = DerivativesProduct(Ix, Ix);
	Iy2 = DerivativesProduct(Iy, Iy);
	
	//// 3. Compute the sum of derivatives' product at each pixel
	//Sobel(Ix2, Sx2, CV_32FC1, 1, 1, ksize, 1, 0, BORDER_DEFAULT);
	//Sobel(Iy2, Sy2, CV_32FC1, 1, 1, ksize, 1, 0, BORDER_DEFAULT);
	//Sobel(Ixy, Sxy, CV_32FC1, 1, 1, ksize, 1, 0, BORDER_DEFAULT);

	// 4. Compute the respone of the detector at each pixel

	R.create(srcImg.rows, srcImg.cols, CV_32FC1);
	float * pRData = (float*)R.data, *pRRow;

	// width là chiều rộng ảnh, height là chiều cao ảnh
	int width = srcImg.cols, height = srcImg.rows;
	// nChannels là số kênh màu
	int nChannels = srcImg.channels();
	// widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = R.step[0];

	float sumIx2, sumIy2, sumIxy, TraceM, detM;

	/*for (int i = 0; i < height; i++, pRData += widthStep)
	{
		pRRow = pRData;
		for (int j = 0; j < width; j++, pRRow += nChannels)
		{
			sumIx2 = sumOfMat(Ix2, blockSize, i, j);
			sumIy2 = sumOfMat(Iy2, blockSize, i, j);
			sumIxy = sumOfMat(Ixy, blockSize, i, j);
			detM = sumIx2 * sumIy2 - sumIxy * sumIxy;
			TraceM = (sumIx2 + sumIy2);
			pRRow[0] = detM - k * TraceM * TraceM;
		}
	}*/

	for (int i = 0; i<height; i++)
		for (int j = 0; j < width; j++)
		{
			sumIx2 = sumOfMat(Ix2, blockSize, i, j);
			sumIy2 = sumOfMat(Iy2, blockSize, i, j);
			sumIxy = sumOfMat(Ixy, blockSize, i, j);
			detM = sumIx2 * sumIy2 - sumIxy * sumIxy;
			TraceM = (sumIx2 + sumIy2);
			R.at<float>(i, j) = detM - k * TraceM * TraceM;
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
