#include "DetectFeaturePoint.h"

int main(int argc, char * argv[])
{
	Mat srcImg = imread("07.jpg", IMREAD_COLOR), GrayImg; //auto load image with BGR
	//int code = atoi(argv[2]);
	//More parameters

	cvtColor(srcImg, GrayImg, COLOR_BGR2GRAY);
	
	Mat R = DetectHarris(GrayImg, 7, 5, 0.05);

	Mat dstImg = ImageWithFeature(srcImg, R, 10000000000.0);

	imshow("Original", srcImg);
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

