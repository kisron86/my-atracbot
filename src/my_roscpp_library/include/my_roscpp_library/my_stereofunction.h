#ifndef MY_STEREOFUNCTION_H
#define MY_STEREOFUNCTION_H

#pragma once
//#include "stdafx.h"


#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "my_stereograb.h"
//#include <cvaux.h>
#include <opencv2/core/types_c.h> //cxtypes.h
//#include <tchar.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
#include <fstream>


#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

void PointCenter(cv::Point center);
void meas_distance(vector<float>& distance);

struct StereoFunction{
  CvMat* _M1;
  CvMat* _M2;
  CvMat* _T;
  CvMat* mx1;
  CvMat* mx2;
  CvMat* my1;
  CvMat* my2;
  CvMat* _Q;			//reprojection matrix
  CvMat* _CamData;

  CvSize imageSize;
  IplImage *img1,
       *img2,
       *img_detect,
       *thres_img,
       *blobs_img,
       *real_disparity;
  CvMat	 *cvma,
       *img1r,
       *img2r,
       *disp,
       *vdisp,
       *pair,
       *depthM;

  void stereoInit(StereoGrab*);
  void stereoCalibration(StereoGrab*);
  void stereoCorrelation(StereoGrab*);
  void stereoSavePointCloud();
  void stereoDetect();

  double reprojectionVars[6];


  int stereoPreFilterSize,
    stereoPreFilterCap,
    stereoDispWindowSize,
    stereoNumDisparities,
    stereoDispTextureThreshold,
    stereoDispUniquenessRatio,
    stereoSavePointCloudValue,
    stereoSaveOriginal;

  int fileNO, threshold, blobArea;
};
#endif // STEREOFUNCTION_H
