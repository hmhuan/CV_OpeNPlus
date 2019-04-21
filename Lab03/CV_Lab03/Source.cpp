#include "DetectFeaturePoint.h"

int main(int argc, char * argv[])
{
	Mat srcImg = imread(argv[1], IMREAD_COLOR), GrayImg, dstImg; //auto load image with BGR
	Mat compImg, compGrayImg;
	Mat R;
	int code = atoi(argv[2]), kSize, k, detector;
	float Threshold, sigma, alpha;
	double result;
	
	// convert ảnh sang ảnh xám.
	cvtColor(srcImg, GrayImg, COLOR_BGR2GRAY);
	

	switch (code) {
	case 1: // grid.jpg 1 13 1000000000 0.05
		kSize = atoi(argv[3]); //17;
		Threshold = atof(argv[4]);//10000000000.0;
		alpha = atof(argv[5]);
		R = DetectHarris(GrayImg, kSize, 5, alpha, Threshold);
		dstImg = ImageWithFeature(srcImg, R, Threshold);
		break;
	case 2:
		dstImg = detectBlob(GrayImg);
		break;
	case 3: //grid.jpg 3 5 1 5
		kSize = atoi(argv[3]);
		sigma = atof(argv[4]);
		k = atof(argv[5]);
		dstImg = detectDOG(GrayImg, kSize, sigma, k);
		break;
	case 4:
		compImg = imread(argv[3]);
		detector = atoi(argv[4]);
		cvtColor(compImg, compGrayImg, COLOR_BGR2GRAY);
		result = matchBySIFT(GrayImg, compGrayImg, detector);
		break;
	default:
		break;
	}

	//

	imshow("Original", srcImg);
	imshow("Gray", GrayImg);
	if (!dstImg.empty())
	{
		cout << "Successful.\n";
		imshow("Image with Features", dstImg);
	}
	else 
	{
		cout << "Failed.\n";
	}
	waitKey(0);
	return 0;
}

//int thresh = 100;
//
//int main()
//{
//	Mat src, gray;
//	// Load source image and convert it to gray
//	src = imread("lena.png", 1);
//	cvtColor(src, gray, CV_BGR2GRAY);
//	Mat dst, dst_norm, dst_norm_scaled;
//	dst = Mat::zeros(src.size(), CV_32FC1);
//	Mat dstImg = src.clone();
//	// Detecting corners
//	cornerHarris(gray, dst, 3, 3, 0.05, BORDER_DEFAULT);
//
//	// Normalizing
//	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//	convertScaleAbs(dst_norm, dst_norm_scaled);
//
//	// Drawing a circle around corners
//	for (int j = 0; j < dst_norm.rows; j++)
//	{
//		for (int i = 0; i < dst_norm.cols; i++)
//		{
//			if ((int)dst_norm.at<float>(j, i) > thresh)
//			{
//				dstImg.at<cv::Vec3b>(j, i)[0] = 0;
//				dstImg.at<cv::Vec3b>(j, i)[1] = 255;
//				dstImg.at<cv::Vec3b>(j, i)[2] = 0;
//			}
//		}
//	}
//
//
//	// Showing the result
//	namedWindow("corners_window", CV_WINDOW_AUTOSIZE);
//	imshow("corners_window", dstImg);
//
//	waitKey(0);
//	return(0);
//}

