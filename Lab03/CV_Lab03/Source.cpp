#include "DetectFeaturePoint.h"

int main(int argc, char * argv[])
{
	Mat srcImg = imread("01.jpg", IMREAD_COLOR), GrayImg; //auto load image with BGR
	//int code = atoi(argv[2]);
	//More parameters

	cvtColor(srcImg, GrayImg, COLOR_BGR2GRAY);
	
	Mat R = DetectHarris(GrayImg, 7, 5, 0.05);

	Mat dstImg = ImageWithFeature(srcImg, R, 10000);

	imshow("Original", srcImg);
	imshow("GrayScale", GrayImg);
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


//int main(int argc, char * argv[])
//{
//	Mat srcImage = imread("06.jpg", 1);
//	Mat dstImage, gray;
//	cvtColor(srcImage, gray, CV_BGR2GRAY);
//	//cornerHarris(gray, dstImage, 2, 3, 0.05);
//	imshow("original", srcImage);
//	imshow("Gray", gray);
//
//	Mat dst, dst_norm, dst_norm_scaled;
//	dst = Mat::zeros(srcImage.size(), CV_32FC1);
//	dstImage = srcImage.clone();
//	// Detecting corners
//	cornerHarris(gray, dst, 5, 7, 0.05, BORDER_DEFAULT);
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
//			if ((int)dst_norm.at<float>(j, i) > 120)
//			{
//				//circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
//				dstImage.at<cv::Vec3b>(j, i)[0] = 0;
//				dstImage.at<cv::Vec3b>(j, i)[1] = 0;
//				dstImage.at<cv::Vec3b>(j, i)[2] = 255;
//			}
//			else
//			{
//				dstImage.at<cv::Vec3b>(j, i)[0] = gray.at<uchar>(j, i);
//				dstImage.at<cv::Vec3b>(j, i)[1] = gray.at<uchar>(j, i);
//				dstImage.at<cv::Vec3b>(j, i)[2] = gray.at<uchar>(j, i);
//			}
//		}
//	}
//	if (dstImage.data != NULL)
//		imshow("dest", dstImage);
//	waitKey(0);
//	return 0;
//}

