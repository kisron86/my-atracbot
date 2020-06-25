#include "my_roscpp_library/my_glcm.h"
#include <ros/ros.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/mat.hpp>

#include <iostream>

using namespace std;
using namespace cv;

void glcm(const Mat img, vector<float> &vec_glcm, bool isShow, bool isPrint)
{
  float energy = 0, contrast = 0, homogenity = 0, IDM = 0, entropy = 0, mean1 = 0;
  int row = img.rows, col = img.cols;
  Mat gl = Mat::zeros(256, 256, CV_32FC1);
  //Mat gl_show = Mat::zeros(256, 256, CV_32FC1);

  //creating glcm matrix with 256 levels,radius=1 and in the horizontal direction
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col - 1; j++)
    {
      gl.at<float>(img.at<uchar>(i, j), img.at<uchar>(i, j + 1)) = gl.at<float>(img.at<uchar>(i, j), img.at<uchar>(i, j + 1)) + 1;
    }
  }

  //show glcm Matrix in image
  if (isShow == true) imshow("GLCM Matrix", gl);
  /*for (int i = 0; i < row; i++)
  {
  for (int j = 0; j < col - 1; j++)
  {
  //int a = gl.at<float>(i, j);
  Point poi(i, j);
  }
  } */
  //imshow("GLCM Matrix_2", gl_show);

  // Print GLCM Matrix
  /*for (int i = 0; i < 256; i++)
  {
  for (int j = 0; j < 256; j++)
  {
  cout << gl.at<float>(i, j) << " ";
  }
  }*/

  // normalizing glcm matrix for parameter determination
  gl = gl + gl.t();
  gl = gl / sum(gl)[0];

  for (int i = 0; i < 256; i++)
  {
    for (int j = 0; j < 256; j++)
    {
      //finding parameters
      energy = energy + gl.at<float>(i, j)*gl.at<float>(i, j);
      contrast = contrast + (i - j)*(i - j)*gl.at<float>(i, j);
      homogenity = homogenity + gl.at<float>(i, j) / (1 + abs(i - j));
      if (i != j)
        IDM = IDM + gl.at<float>(i, j) / ((i - j)*(i - j));  //Taking k=2;
      if (gl.at<float>(i, j) != 0)
        entropy = entropy - gl.at<float>(i, j)*log10(gl.at<float>(i, j));
      mean1 = mean1 + 0.5*(i*gl.at<float>(i, j) + j*gl.at<float>(i, j));
    }
  }
  vec_glcm.push_back(energy);
  vec_glcm.push_back(contrast);
  vec_glcm.push_back(homogenity);
  vec_glcm.push_back(IDM);
  vec_glcm.push_back(entropy);
  vec_glcm.push_back(mean1);

  /*  for (int i = 0; i<256; i++)
  {
  for (int j = 0; j<256; j++)
  cout << a[i][j] << "\t";
  cout << endl;
  }*/
  if (isPrint == true) {
    cout << "energy=" << energy << endl;
    cout << "contrast=" << contrast << endl;
    cout << "homogenity=" << homogenity << endl;
    cout << "IDM=" << IDM << endl;
    cout << "entropy=" << entropy << endl;
    cout << "mean=" << mean1 << endl;
  }

}

