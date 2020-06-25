#include <ros/ros.h>
#include "my_roscpp_library/my_super_roscpp_library.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

using namespace std;
using namespace cv;

std::string cascadeName = "/home/kisron/catkin_workspace/FeatureXML/cascade_waste.xml";
cv::CascadeClassifier cascade;
double scale = 1;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "video_cam");
    ros::NodeHandle nh;

	cv::VideoCapture cap(4);
	cv::Mat frame;
	int i;

    for(;;)
    {
		cap >> frame;

		if (frame.empty())
			break;

		vector<cv::Rect> gopro;
		cv::Mat small_img_gray, small_img(cvRound(frame.rows / scale), cvRound(frame.cols / scale), CV_8UC1);
		cv::Point center;

		if (!cascade.load(cascadeName))
		{
			printf("--(!)Error loading cascade, please change cascade_name in source code.\n");
			return 0;
		};

		cv::resize(frame, small_img, small_img.size(), 0, 0, 1);
		cvtColor(small_img, small_img_gray, CV_BGR2GRAY);
		cv::equalizeHist(small_img_gray, small_img_gray);

		cascade.detectMultiScale(small_img_gray, gopro, 1.5, 4.5, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(24 / scale, 24 / scale));

		for (i = 0; i < gopro.size(); i++)
			cv::rectangle(frame, cvPoint(cvRound(gopro[0].x)*scale, cvRound(gopro[0].y)*scale), cvPoint((cvRound(gopro[0].x + gopro[0].width)*scale), (cvRound(gopro[0].y + gopro[0].height)*scale)), cv::Scalar(255,0 , 0), 2, 8, 0);

		cv::imshow("Object Detection", frame);

		cv::waitKey(10);  
        if(waitKey(30) == 27){ break; }   //27 ASCII Esc  
        }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
