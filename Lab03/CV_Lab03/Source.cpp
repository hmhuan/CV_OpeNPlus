#include "DetectFeaturePoint.h"

int main(int argc, char * argv[])
{
	Mat srcImg = imread(argv[1], IMREAD_COLOR), GrayImg, dstImg; //auto load image with BGR
	Mat compImg, compGrayImg;
	Mat R;
	int code = atoi(argv[2]), kSize, blockSize, k, detector;
	float Threshold, sigma, alpha;
	double result;
	
	// convert ảnh sang ảnh xám.
	cvtColor(srcImg, GrayImg, COLOR_BGR2GRAY);
	

	switch (code) {
	case 1: // grid.jpg 1 13 1000000000 0.05 | lena.png 1 7 1000000000.0 0.05
		blockSize = atoi(argv[3]); //17;
		Threshold = atof(argv[4]);//10000000000.0;
		alpha = atof(argv[5]);
		R = DetectHarris(GrayImg, blockSize, 5, alpha, Threshold);
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
		imwrite("res_10.jpg", dstImg);
	}
	else 
	{
		cout << "Failed.\n";
	}
	waitKey(0);
	return 0;
}