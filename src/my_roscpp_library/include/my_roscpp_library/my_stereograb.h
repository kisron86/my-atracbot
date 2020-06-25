#ifndef MY_STEREOGRAB_H
#define MY_STEREOGRAB_H

//stereograb.h file header ketika mendefinisi
// untuk variable dan fungsi
#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/mat.hpp>

#define WIDTH 320
#define HEIGHT 240


struct StereoGrab{

  void stereoGrabInitFrames();
  void stereGrabFrames();
  void stereoGrabStopCam();
  IplImage* imageLeft;
  IplImage* imageRight;

};

#endif // STEREOGRAB_H
