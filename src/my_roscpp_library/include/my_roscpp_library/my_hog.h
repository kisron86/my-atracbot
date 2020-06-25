#ifndef MY_HOG_H
#define MY_HOG_H

#include <ros/ros.h>
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/mat.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void ini_coba();
void computeMagAngle(InputArray src, OutputArray mag, OutputArray ang);

void computeHOG(InputArray mag, InputArray ang, OutputArray dst, int dims, bool isWeighted);

void featureVecFullPrint(vector<float>& vHOG, int loop, bool print);

void initVFile(int n);

void saveFeatureVecFull(char SaveHogDesFileName[100]);

void saveHOGglcmVec(char SaveHogglcmDesFileName[100]);

void copyHOG_GLCMtoVec(vector<float>& vec_glcm, vector< vector <float>>& vecForSvm);

void reduceFeatureUsingPCA(Mat reduced, int maxComp, bool isPrint);

void reduceFeatureUsingPCAinSVM(Mat reduced, Mat Output, vector<float>& vectorHogGlcm, bool isPrint);

void saveEigenValues(char SaveHogDesFileName[100]);

Mat get_hogdescriptor_visual_image(Mat& origImg,
  vector< float>& descriptorValues,
  Size winSize,
  Size cellSize,
  int scaleFactor,
  double viz_factor);

#endif // MY_HOG_H
