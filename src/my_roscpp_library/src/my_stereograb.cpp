#include "my_roscpp_library/my_stereograb.h"

CvCapture *capture1=NULL, *capture2=NULL;

void StereoGrab::stereoGrabInitFrames(){

  capture1=cvCaptureFromCAM(0);
  assert(capture1!=NULL);
  cvWaitKey(100);
  capture2=cvCaptureFromCAM(2);
  assert(capture2!=NULL);

  cvSetCaptureProperty(capture1,CV_CAP_PROP_FRAME_WIDTH,WIDTH);
  cvSetCaptureProperty(capture1,CV_CAP_PROP_FRAME_HEIGHT,HEIGHT);
  cvSetCaptureProperty(capture2,CV_CAP_PROP_FRAME_WIDTH,WIDTH);
  cvSetCaptureProperty(capture2,CV_CAP_PROP_FRAME_HEIGHT,HEIGHT);
}

void StereoGrab::stereGrabFrames(){
	imageLeft = cvQueryFrame(capture1);
	imageRight = cvQueryFrame(capture2);
}

void StereoGrab::stereoGrabStopCam(){
  cvReleaseCapture( &capture1 );
  cvReleaseCapture( &capture2 );
}
