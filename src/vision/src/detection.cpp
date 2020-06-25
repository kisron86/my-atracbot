#include <ros/ros.h>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <time.h> // to calculate time needed
#include <limits.h> // to get INT_MAX, to protect against overflow

#include <iomanip>
#include <iostream>
#include <stdlib.h>
//#include <Windows.h>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include <sstream>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/mat.hpp>

#include "my_roscpp_library/my_stereofunction.h"
#include "my_roscpp_library/my_stereograb.h"
#include "my_roscpp_library/my_glcm.h"
#include "my_roscpp_library/my_hog.h"

using namespace std;
using namespace cv;

vector < vector <float> > vecSVM;
CvSVM svm;

String waste_cascade_name = "/home/kisron/catkin_workspace/FeatureXML/waste.xml";
CascadeClassifier waste_cascade;

Mat image;
Mat crop_image;
/// Parameter Feature Extraction
Size sizes(32, 32);
int nFile = 50; // n-file to extract
//string folder = "D:\\Project\\TA\\Data_Trainer\\Clip_Images\\DataTraining\\Data_Testing\\";  // folder input 
string folder = "/home/kisron/catkin_workspace/Data_Testing";  // folder input 
Size cutRoi(8, 8);
char buffer[1000];
int counter = 0, imcount = 0;
Mat imgPar;
vector<float> vHOG;
vector<float> vec_glcm;
vector<float> vectorHogGlcm;
Mat pre_crop_image;
Mat reduced(1, 144, CV_32F);
Mat imgToSvm(1, 12, CV_32F);

int flag_class_result = 0, flag_print = 0, flag_bukan_sampah = 1;
int ppp = 0, nnn = 0, net = 0;
String className;
Mat ClassImg;
Point track_roi_top; Point track_roi_bot;
Point roi_center;

IplImage* imageLeft;
IplImage* imageRight;

CvCapture *capture1=NULL, *capture2=NULL;

Mat frame_cpy;

void Classifier() {
	///Classification whether data is positive, negative or non sampah
	int result = svm.predict(imgToSvm);
	if (flag_print == 1) {
		cout << "Classification..." << " ";
		cout << "result: " << result << endl;
	}

	///Count data
	if (result < 10) {
		ppp++;
		className = "#Organik";
		flag_bukan_sampah = 1;
	}
	else if (result < 29) {
		nnn++;
		className = "#Non Organik";
		flag_bukan_sampah = 1;
	}
	else if (result > 28) {
		net++;
		className = "#Bukan Sampah";
		flag_bukan_sampah = 0;
	}

	Point p_atas(5, 15);
	/// Draw class in Image
	if (flag_class_result == 1) {
		putText(ClassImg, className, track_roi_top, FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 255), 2);
		if(flag_bukan_sampah == 1)
			arrowedLine(ClassImg, Point(160, 240), roi_center, Scalar(220, 20, 60), 2, 8, 0);
	}

	///Write image
	/*sprintf_s(buffer, "D:\\Project\\TA\\Data_Trainer\\Clip_Images\\DataTraining\\Hasil_Testing\\Test%u.png", nameCount++);
	//imwrite(buffer, ClassImg);
	//printf(" positive/negative/netral = (%d/%d/%d) \n", ppp, nnn, net); */
}

void ExtractFeature(Mat capt, int looper) {
	///load images
	image = crop_image.clone();
	cvtColor(image, image, COLOR_BGR2GRAY);
	resize(image, image, sizes);
	if (image.empty()) cout << "No image loaded" << endl;

	/// begin parsing image to 16 cell 8*8
	for (int i = 0; i < image.rows; i += cutRoi.height) {
		for (int j = 0; j < image.cols; j += cutRoi.width) {
			Rect roi(i, j, cutRoi.width, cutRoi.height);
			Mat crop = image(roi);
			sprintf(buffer, "/home/kisron/catkin_workspace/ROI/%u.png", counter++);
			imwrite(buffer, crop);
		}
	}
	counter = 0;

	/// read the parsing images
	for (int x = 0; x < 16; x++) {
		string folderROI = "/home/kisron/catkin_workspace/ROI/";	// read folder
		string suffix = ".png";	// file type
		stringstream ss;
		ss << setw(0) << setfill('0') << counter++; // 0000, 0001, 0002, etc...
		string number = ss.str();

		string name = folderROI + number + suffix;
		imgPar = imread(name);
		if (imgPar.empty()) cout << "No parsing image loaded...!" << endl;

		/// Extract Feature using HOG
		Mat mag, ang;
		computeMagAngle(imgPar, mag, ang);

		Mat wHogFeature;
		computeHOG(mag, ang, wHogFeature, 9, true);
	}
	counter = 0;
	featureVecFullPrint(vHOG, looper, false);

	/// Extract Feature using glcm
	imshow("image", image);
	glcm(image, vec_glcm, false, false);
	waitKey(1);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "detection");
    ros::NodeHandle nh;

    FileStorage read_FeatureXml("/home/kisron/catkin_workspace/FeatureXML/eigenValues_All_2.xml", FileStorage::READ);
	if (read_FeatureXml.isOpened()) cout << "HOGSampah xml loaded" << endl;

    ///Feature Mat
	Mat eMat;
	read_FeatureXml["Descriptor_of_images"] >> eMat;
	read_FeatureXml.release();

	/// Load trained SVM xml data
	svm.load("/home/kisron/catkin_workspace/FeatureXML/Svm_Feature_Linear_All_2.xml");
	cout << "SVM Feature Linear All Loaded" << endl;

    //  -- 1. Load the cascades
	if (!waste_cascade.load(waste_cascade_name)) { printf("--(!)Error loading xml\n"); };
	cout << "begin lur" << endl;

    capture1=cvCaptureFromCAM(2);
    assert(capture1!=NULL);
    cvWaitKey(100);
    capture2=cvCaptureFromCAM(4);
    assert(capture2!=NULL);

    cvSetCaptureProperty(capture1,CV_CAP_PROP_FRAME_WIDTH,WIDTH);
    cvSetCaptureProperty(capture1,CV_CAP_PROP_FRAME_HEIGHT,HEIGHT);
    cvSetCaptureProperty(capture2,CV_CAP_PROP_FRAME_WIDTH,WIDTH);
    cvSetCaptureProperty(capture2,CV_CAP_PROP_FRAME_HEIGHT,HEIGHT);
    
    imageLeft = cvQueryFrame(capture1);
    imageRight = cvQueryFrame(capture2);

    cout << "Stop" << endl;

    Mat read_cam_left = imread("/home/kisron/catkin_workspace/images/cam_left.jpg");
	frame_cpy = read_cam_left.clone();
	//enchanceImage = read_cam_left.clone();
	//cv::cvtColor(enchanceImage, enchanceImage, CV_BGR2GRAY);
	//cv::blur(enchanceImage, enchanceImage, Size(3, 3));
	pre_crop_image = frame_cpy.clone();
	//res = frame_cpy.clone();

    ExtractFeature(pre_crop_image, 0); /// Feature Extraction
	reduceFeatureUsingPCAinSVM(eMat, reduced, vectorHogGlcm, false); /// Reduce HOG using PCA
	for (auto i = vec_glcm.begin(); i != vec_glcm.end(); ++i) vectorHogGlcm.push_back(*i);
	    int no = 0;
	    for (auto i = vectorHogGlcm.begin(); i != vectorHogGlcm.end(); ++i) imgToSvm.at<float>(0, no++) = *i;
	    vec_glcm.clear(); vectorHogGlcm.clear(); /// Empty the vector
		Classifier();
		//print_distance_angle();
}