#include "ros/ros.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "talker");
  ros::NodeHandle n;

  ros::Rate loop_rate(10);

  int count = 0;
  while (ros::ok())
  {
    VideoCapture capr(2);  // kamera kanan
    capr.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    capr.set(CV_CAP_PROP_FRAME_WIDTH, 320);

    if(!capr.isOpened()) {
      cout << "Cannot open the video file. \n";
      return -1;
  }

  namedWindow("Kamera_Kanan",CV_WINDOW_AUTOSIZE);
  moveWindow("Kamera_Kanan",725,200);

  Mat framer;
  while(1){
    if (!capr.read(framer)){
      cout<<"\n Cannot read the video file. \n";
      break;
    }
    imshow("Kamera_Kanan", framer);
    if(waitKey(30) == 27){ break; }   //27 ASCII Esc
  }
  }
  return 0;
}

