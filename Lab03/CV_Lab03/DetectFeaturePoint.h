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
				 circle(dstImg, Point(j, i), 3.0, Scalar(0, 0, 255), 2, 8);
			}
		}	
	}
	return dstImg;
}


/* Non maximum suppression
Hàm loại bỏ các điểm không phải cực trị cục bộ
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
			float currVal = R.at<float>(i, j); // Xét giá trị tại (i, j).
			bool check = true;
			if (currVal > Threshold)
			{
				// Duyệt qua block
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
					Nms.at<uchar>(i, j) = 1; // Nếu lớn hơn ngưỡng và cực đại cục bộ thì giữ lại.
				else
					Nms.at<uchar>(i, j) = 0; // Nếu không phải cực đại cục bộ thì loại bỏ.
			}
			else
				Nms.at<uchar>(i, j) = 0; // Ngược lại nếu nhỏ hơn ngưỡng thì loại bỏ.
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
			mul = Ix.at<float>(i, j) * Iy.at<float>(i, j) * 1.0f;
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
Mat DetectHarris(const Mat& srcImg, int blockSize, int ksize, float k, float & Threshold)
{
	// Init @sobelX and @sobelY image
	Mat blurImg;
	Mat Ix, Iy;
	Mat Ix2, Iy2, Ixy;
	Mat R, NormR, ScaledR;
	
	// 1. Lọc ảnh với Gaussian để giảm nhiễu
	GaussianBlur(srcImg, blurImg, cv::Size(ksize, ksize), 0, 0, BORDER_DEFAULT);
	
	// 2. Compute x and y derivatives of @srcImg sobel 3 x 3
	float sobelX[9] = {1.0/4, 0, -1.0/4, 1.0/2, 0, -1.0/2, 1.0/4, 0, -1.0/4};
	float sobelY[9] = {-1.0/4, -1.0/2, -1.0/4, 0, 0, 0, 1.0/4, 1.0/2, 1.0/4};
	vector<float> kernelX, kernelY;
	for (int i = 0; i < 9; i++)
	{
		kernelX.push_back(sobelX[i]);
		kernelY.push_back(sobelY[i]);
	}
	Ix = convolve(blurImg, kernelX, 3); // tích chập ảnh đã lọc gaussian với kernel sobelX
	Iy = convolve(blurImg, kernelY, 3); // tích chập ảnh đã lọc gaussian với kernel sobelY
	// 3. Compute products of derivatives at every pixel
	Mat GIxy, GIx2, GIy2;
	Ixy = DerivativesProduct(Ix, Iy);
	Ix2 = DerivativesProduct(Ix, Ix);
	Iy2 = DerivativesProduct(Iy, Iy);

	// 3. Gaussian các ma trận ảnh Ix2, Iy2, Ixy vừa tìm được
	GaussianBlur(Ixy, GIxy, Size(ksize, ksize), 1, 1, BORDER_DEFAULT);
	GaussianBlur(Ix2, GIx2, Size(ksize, ksize), 1, 1, BORDER_DEFAULT);
	GaussianBlur(Iy2, GIy2, Size(ksize, ksize), 1, 1, BORDER_DEFAULT);

	// 4. Compute the Respone of the detector at each pixel
	R.create(srcImg.rows, srcImg.cols, CV_32F);

	// width là chiều rộng ảnh, height là chiều cao ảnh
	int width = srcImg.cols, height = srcImg.rows;

	float sumIx2, sumIy2, sumIxy, TraceM, detM, lamda1, lamda2;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			sumIx2 = sumOfMat(GIx2, blockSize, i, j);  // Tính sum Ix2 trong blockSize
			sumIy2 = sumOfMat(GIy2, blockSize, i, j);  // Tính sum Iy2 trong blockSize
			sumIxy = sumOfMat(GIxy, blockSize, i, j);  // Tính sum Ixy trong blockSize
			detM = (sumIx2 * sumIy2) - (sumIxy * sumIxy);
			TraceM = sumIx2 + sumIy2;
			R.at<float>(i, j) = detM - k * TraceM * TraceM;
		}
	// In ra min - max để test ngưỡng
	double min, max;
	cv::minMaxLoc(R, &min, &max);
	cout << max << " " << min << "\n";
	
	// 5. Non maximum suppression với ma trận Respond R
	Mat Nms = NonMaximumSuppression(R, blockSize, Threshold);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			if (Nms.at<uchar>(i, j) == 0)
				R.at<float>(i, j) = 0;
		}
	return R; // Trả về ma trận Respond
}

/*LoG - Laplacian Of Gaussian.
blockSize cho bằng 5.
Input:
	- image: ảnh nguồn.
	- sigma: giá trị sigma.
	- kSize: kích thước của sổ.
Output:
	- LoGImg: ảnh sau khi áp dụng LoG.
*/
Mat LoG(Mat & srcImage, float sigma, int kSize)
{
	// Nguồn:
	// https://academic.mu.edu/phys/matthysd/web226/Lab02.htm
	Mat LoGImg;
	int height = srcImage.rows, width = srcImage.cols;
	vector<float> kernel; //kernel (kSize x kSize)
	int halfkSize = kSize / 2;
	int n = kSize * kSize; // Kích thước kernel

	// khởi tạo kernel
	float sum = 0.0f; // Tổng của kernel
	float sigma2 = 2 * sigma * sigma; //2 * sigma^2
	float ms = M_PI * sigma2 * sigma2; // mẫu số pi * sigma^4
	for (int y = -halfkSize; y <= halfkSize; y++)
		for (int x = -halfkSize; x <= halfkSize; x++)
		{
			float temp = (x*x + y*y)/(2*sigma2);
			float h = -(1 - temp)*expf(-temp) / ms;
			sum += h;
			kernel.push_back(h);
		}
	//Chuẩn hóa kernel 
	for (int i = 0; i < n; i++)
		kernel[i] /= sum;
	LoGImg = convolve(srcImage, kernel, kSize);
	return LoGImg;
}

/* Blob use Laplacian of Gaussian

*/
Mat detectBlob(Mat & srcImage, int kSize)
{
	// Tham khảo source: 
	// https://github.com/Abhishekmamidi123/Computer-Vision/blob/master/Assignment-1/3_Blob_detector.py
	// Xét scale với 3 sigma
	vector<Mat> imageSigma;
	Mat blob;
	float sigma[3] = { 1, 2.2, 3.4}; // Danh sách sigma để áp dụng LoG
	int dx[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	int dy[9] = { -1, 0, 1, -1, 0, 1 , -1, 0, 1 };
	int n = 3;
	int height = srcImage.rows, width = srcImage.cols;
	bool check;
	// ảnh mờ có thể áp dụng bộ lọc Laplacian
	for (int i = 0; i < n; i++)
	{
		imageSigma.push_back(LoG(srcImage, sigma[i], kSize));
	}
	
	blob.create(height, width, CV_32F);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			check = true;
			float currVal = imageSigma[1].at<float>(i, j); // Lấy tại scale trung gian giá trị pixel.
			for (int k = 0; k < 9; k++)
				if (isInRange(i + dx[k], j + dy[k], height, width))
				{
					if (imageSigma[0].at<float>(i + dx[k], j + dy[k]) > currVal)
					{
						check = false;
						break;
					}
					if (imageSigma[2].at<float>(i + dx[k], j + dy[k]) > currVal)
					{
						check = false;
						break;
					}
				}
			if (check)
				blob.at<float>(i, j) = 1;
			else
				blob.at<float>(i, j) = 0;
		}
	return blob;
}
/* Different of Gaussian
Input:
	- srcImage: Ảnh nguồn.
	- kSize: kích thước của sổ.
	- sigma: giá trị sigma 1.
	- k: hệ số chênh lệch giữa 2 sigma.
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
