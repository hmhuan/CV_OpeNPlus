#pragma once
#define _USE_MATH_DEFINES
#include "Convolution.h"
#include  <math.h>

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
				 //pRow[0] = 0; // Blue = 0
				// pRow[1] = 0; // Green = 0
				 //pRow[2] = 255; // Red = 255
				 circle(dstImg, Point(i, j), 3.0, Scalar(0, 0, 255), 2, 8);
			}
		}	
	}
	return dstImg;
}


/*

*/
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
	float mul;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			mul = Ix.at<uchar>(i, j) * Iy.at<uchar>(i, j) * 1.0f;
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
*	@ksize: aperture parameter of GaussianBlur used
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
	GaussianBlur(srcImg, blurImg, cv::Size(ksize, ksize), 1);
	
	// 2. Compute x and y derivatives of @srcImg sobel 3 x 3
	Sobel(blurImg, Ix, CV_8UC1, 1, 0, 3);
	Sobel(blurImg, Iy, CV_8UC1, 0, 1, 3);

	// 3. Compute products of derivatives at every pixel
	Mat GIxy, GIx2, GIy2;
	Ixy = DerivativesProduct(Ix, Iy);
	Ix2 = DerivativesProduct(Ix, Ix);
	Iy2 = DerivativesProduct(Iy, Iy);

	// 3. Compute the sum of derivatives' product at each pixel
	/*Sobel(Ix2, GIx2, CV_32F, 1, 1, ksize, 1, 0, BORDER_DEFAULT);
	Sobel(Iy2, GIy2, CV_32F, 1, 1, ksize, 1, 0, BORDER_DEFAULT);
	Sobel(Ixy, GIxy, CV_32F, 1, 1, ksize, 1, 0, BORDER_DEFAULT);*/

	GaussianBlur(Ixy, GIxy, Size(ksize, ksize), 2.5);
	GaussianBlur(Ix2, GIx2, Size(ksize, ksize), 2.5);
	GaussianBlur(Iy2, GIy2, Size(ksize, ksize), 2.5);
	// 4. Compute the respone of the detector at each pixel
	R.create(srcImg.rows, srcImg.cols, CV_32F);

	// width là chiều rộng ảnh, height là chiều cao ảnh
	int width = srcImg.cols, height = srcImg.rows;

	float sumIx2, sumIy2, sumIxy, TraceM, detM, lamda1, lamda2;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			sumIx2 = sumOfMat(GIx2, blockSize, i, j);
			sumIy2 = sumOfMat(GIy2, blockSize, i, j);
			sumIxy = sumOfMat(GIxy, blockSize, i, j);
			//lamda1 = (sumIx2 + sumIy2 + sqrt(4 * sumIxy*sumIxy + (sumIx2 - sumIy2) * (sumIx2 - sumIy2)));
			//lamda2 = (sumIx2 + sumIy2 - sqrt(4 * sumIxy*sumIxy + (sumIx2 - sumIy2) * (sumIx2 - sumIy2)));
			//sumIx2 = GIx2.at<float>(i, j);
			//sumIy2 = GIy2.at<float>(i, j);
			//sumIxy = GIxy.at<float>(i, j);
			detM = sumIx2 * sumIy2 - sumIxy * sumIxy;
			TraceM = sumIx2 + sumIy2;
			R.at<float>(i, j) = detM - k * TraceM * TraceM;
		}

	Mat Nms = NonMaximumSuppression(R, blockSize, Threshold);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			if (Nms.at<uchar>(i, j) == 0)
				R.at<float>(i, j) = 0;
		}
	return R;
}

/* Laplacian of Gaussian

*/
Mat detectBlob(Mat image)
{


	Mat dstImage;


	return dstImage;
}
/* Different of Gaussian
Input:
	-
Output:
	ảnh kết quả detect bằng DoG
*/
Mat detectDOG(Mat & srcImage, int kSize, float sigma, float k)
{
	Mat dstImage;
	int height = srcImage.rows, width = srcImage.cols;
	vector<float> kernel;
	int halfkSize = kSize / 2;
	int n = kSize * kSize;

	// khởi tạo kernel
	float sum = 0.0f; // Tổng của kernel
	float sigma2 = 2 * sigma * sigma; //2 * sigma bình phương
	float k2 = k * k; // k bình phương
	float ms = 2 * M_PI * sigma2; // mẫu số
	for (int y = -halfkSize; y <= halfkSize; y++)
		for (int x = -halfkSize; x <= halfkSize; x++)
		{
			float h = expf(-(y * y + x * x) / sigma2) / ms - expf(-(y * y + x * x) / (k2 * sigma2)) / (ms * k2);
			sum += h;
			kernel.push_back(h);
		}
	//Chuẩn hóa kernel 
	for (int i = 0; i < n; i++)
		kernel[i] /= sum;
	dstImage = convolve(srcImage, kernel, kSize);
	return dstImage;
}
/* SIFT

*/
double matchBySIFT(Mat image1, Mat image2, int detector)
{
	
	return 0;
}
