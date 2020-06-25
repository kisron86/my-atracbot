#include <ros/ros.h>

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

Mat mag, ang;
char buffer[1000];
char buffer1[200];
Mat imgPar;
Mat reduced(1, 144, CV_32F);
int counter = 0, imcount = 0;
Size cutRoi(8, 8);
Mat imgToSvm(1, 12, CV_32F);
String className;
vector<float> vHOG;
vector<float> vec_glcm;
vector<float> vectorHogGlcm;
Mat ClassImg;
Mat image;
Mat pre_crop_image;
Mat crop_image;
int thresholdC = 100;
Point roi_center;
vector<float> meas_dist;
int flag_class_result = 0, flag_print = 0, flag_bukan_sampah = 1;
int f_counter = 0;

vector < vector <float> > vecSVM;
CvSVM svm;

/// Parameter Feature Extraction
Size sizes(32, 32);
int nFile = 50; // n-file to extract
//string folder = "D:\\Project\\TA\\Data_Trainer\\Clip_Images\\DataTraining\\Data_Testing\\";  // folder input 
string folder = "/home/kisron/catkin_workspace/Data_Testing";  // folder input 

string suffix = ".jpg";
int lCount = 0;
int nameCount = 0;
int ppp = 0, nnn = 0, net = 0;

/// Parameter Object Detection
String waste_cascade_name = "/home/kisron/catkin_workspace/FeatureXML/waste.xml";
CascadeClassifier waste_cascade;
string window_name_detect = "Capture - waste detection";
Mat gray;
int fps_counter = 0, fps_max = 0;
RNG rng(12345);
vector<Rect> waste;
Point roi_top; Point roi_bot;
Mat frame_cpy;
int run = 0;

/// Parameter Tracking
Mat res;
bool found = false;
double ticks = 0;
int notFoundCount = 0;
int initialize = 0;
Point track_roi_top; Point track_roi_bot;

/// Parameter enchaneTracking
int detect_roi_top = 0;
int detect_roi_bot = 0;
int detect_roi_left = 0;
int detect_roi_right = 0;
Mat enchanceImage;

// >>>> Kalman Filter
int stateSize = 6;
int measSize = 4;
int contrSize = 0;
unsigned int type = CV_32F;
KalmanFilter kf(stateSize, measSize, contrSize, type);

Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]
								//Mat procNoise(stateSize, 1, type)
								// [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

								// Transition State Matrix A
								// Note: set dT at each processing step!
								// [ 1 0 dT 0  0 0 ]
								// [ 0 1 0  dT 0 0 ]
								// [ 0 0 1  0  0 0 ]
								// [ 0 0 0  1  0 0 ]
								// [ 0 0 0  0  1 0 ]
								// [ 0 0 0  0  0 1 ]
								// <<<< Kalman Filter

#define CALIBRATION 0

StereoGrab* grab= new StereoGrab();
StereoFunction* stereoFunc = new StereoFunction();

void webcam1(){
	VideoCapture capr(2), capl(4);
	//reduce frame size
	capl.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	capl.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	capr.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	capr.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	namedWindow("Left");
	namedWindow("Right");
	while (1){
	
		Mat camera1,camera2;
		capl.read(camera1);
		capr.read(camera2);
		cvtColor(camera1, camera1, COLOR_BGR2GRAY);
		cvtColor(camera2, camera2, COLOR_BGR2GRAY);
		imshow("Left", camera1);
		imshow("Right", camera2);
		if (waitKey(30) >= 0) break;
	}
	capl.release();
	capr.release();
	
}

/////////////////////------------CAPTURE IMAGE-------------////////////////////////////
void capture(){

	VideoCapture capr(2), capl(4);
	//reduce frame size
	capl.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	capl.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	capr.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	capr.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	namedWindow("Left");
	namedWindow("Right");
	cout << "Tekan C simpan gambar ..." << endl;
	char choice = 'z';
	int count = 0;
		while(choice != 'q') {
		//grab frames quickly in succession
		capl.grab();
		capr.grab();
		//execute the heavier decoding operations
		Mat framel, framer;
		capl.retrieve(framel);
		capr.retrieve(framer);

		cvtColor(framel, framel, COLOR_BGR2GRAY);
		cvtColor(framer, framer, COLOR_BGR2GRAY);

			if(framel.empty() || framer.empty()) break;
				imshow("Left", framel);
				imshow("Right", framer);
			if(choice == 'c') {
		//save files at proper locations if user presses 'c'
				stringstream l_name, r_name;
				l_name << "left" << setw(2) << setfill('0') << count << ".jpg";
				r_name << "right" << setw(2) << setfill('0') << count << ".jpg";
				imwrite( l_name.str(), framel);
				imwrite( r_name.str(), framer);
				cout << "Saved set " << count << endl;
				count++;
			}
		choice = char(waitKey(1));
		}
	capl.release();
	capr.release();
}

void view_camera(){
	CvSize imageSize = { 0, 0 };

	grab->stereoGrabInitFrames();
	grab->stereGrabFrames();
	
	IplImage *frame1 = grab->imageLeft;
	IplImage *frame2 = grab->imageRight;
	IplImage *immg1, *immg2;

	immg1 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
	immg2 = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

	char choice = 'z';
	while(choice!='q'){

		grab->stereGrabFrames();
		frame1 = grab->imageLeft;
		frame2 = grab->imageRight;

		cvCvtColor(grab->imageLeft, immg1, CV_RGB2GRAY);
		cvCvtColor(grab->imageRight, immg2, CV_RGB2GRAY);

		cvShowImage("camera left", immg1);
		cvShowImage("camera right", immg2);
			
		if(cvWaitKey(15)==27) break;
	}
}

void playcal(){

	stereoFunc->stereoCalibration(grab);
	
}

void loadcorrelation(){
	ifstream params;
	params.open("/home/kisron/catkin_workspace/Correlation/Correlation.txt");
	if(!params){
		cout << "file gak isok!";
		exit(1);
	}else{
		string temp;
		int value;
		while(params>>temp){
		params>>value;
		stereoFunc->stereoDispWindowSize = value;
		params>>temp;
		params>>value;
		stereoFunc->stereoDispTextureThreshold = value;
		params>>temp;
		params>>value;
		stereoFunc->stereoDispUniquenessRatio = value;
		params>>temp;
		params>>value;
		stereoFunc->stereoNumDisparities = value;
		params>>temp;
		params>>value;
		stereoFunc->threshold = value;
		params>>temp;
		params>>value;
		stereoFunc->blobArea = value;
		}
		stereoFunc->stereoPreFilterSize = 63;
		stereoFunc->stereoPreFilterCap = 12;//32;//63; 
		stereoFunc->stereoSavePointCloudValue = 0;

		params.close();
	}

}

void onWindowBarSlide(int pos)
{
	stereoFunc->stereoDispWindowSize = cvGetTrackbarPos("SADSize", "Stereo Controls");
	if(stereoFunc->stereoDispWindowSize < 5)
		{	stereoFunc->stereoDispWindowSize = 5;
			stereoFunc->stereoCorrelation(grab);
		}	
	else if ( stereoFunc->stereoDispWindowSize%2 == 0) 	  
		{
			stereoFunc->stereoDispWindowSize += 1;
			stereoFunc->stereoCorrelation(grab);
		}
	else stereoFunc->stereoCorrelation(grab); 
}

void onTextureBarSlide(int pos){
	stereoFunc->stereoDispTextureThreshold = cvGetTrackbarPos("Texture th", "Stereo Controls");
	if(stereoFunc->stereoDispTextureThreshold) 
		stereoFunc->stereoCorrelation(grab);
}

void onUniquenessBarSlide(int pos){
	stereoFunc->stereoDispUniquenessRatio = cvGetTrackbarPos("Uniqueness", "Stereo Controls");
	if(stereoFunc->stereoDispUniquenessRatio>=0)
		stereoFunc->stereoCorrelation(grab);
}

void onNumDisparitiesSlide(int pos){
	stereoFunc->stereoNumDisparities = cvGetTrackbarPos("Num.Disp", "Stereo Controls");
	while(stereoFunc->stereoNumDisparities%16!=0 || stereoFunc->stereoNumDisparities==0)
		stereoFunc->stereoNumDisparities++;

	stereoFunc->stereoCorrelation(grab);
}

void onPreFilterSizeBarSlide(int pos){
	stereoFunc->stereoPreFilterSize = cvGetTrackbarPos("PrFil.Size", "Stereo Controls");
	if(stereoFunc->stereoPreFilterSize>=5)
		if(stereoFunc->stereoPreFilterSize%2!=0)
				stereoFunc->stereoCorrelation(grab);
		else {
				++(stereoFunc->stereoPreFilterSize);
				stereoFunc->stereoCorrelation(grab);}
	else {
				stereoFunc->stereoPreFilterSize = 5;
				stereoFunc->stereoCorrelation(grab);}
		
} 

void onPreFilterCapBarSlide(int pos){
	stereoFunc->stereoPreFilterCap = cvGetTrackbarPos("PrFil.Cap", "Stereo Controls");
	if(stereoFunc->stereoPreFilterCap == 0) 
		{	stereoFunc->stereoPreFilterCap = 1;
			stereoFunc->stereoCorrelation(grab);
		}
	else if( stereoFunc->stereoPreFilterCap > 63)		
		{	stereoFunc->stereoPreFilterCap = 63;
			stereoFunc->stereoCorrelation(grab);
		}
	else 	stereoFunc->stereoCorrelation(grab);
}

void stereoCorrelationControl(){
	cvNamedWindow("Stereo Controls",0);
	cvResizeWindow("Stereo Controls", 350,	350);
	cvCreateTrackbar("SADSize", "Stereo Controls", &stereoFunc->stereoDispWindowSize,255, onWindowBarSlide);
	cvCreateTrackbar("Uniqueness", "Stereo Controls", &stereoFunc->stereoDispUniquenessRatio,25, onUniquenessBarSlide);
	cvCreateTrackbar("PrFil.Size", "Stereo Controls", &stereoFunc->stereoPreFilterSize,101, onPreFilterSizeBarSlide);
	cvCreateTrackbar("PrFil.Cap", "Stereo Controls", &stereoFunc->stereoPreFilterCap,63, onPreFilterCapBarSlide);
	cvCreateTrackbar("Num.Disp", "Stereo Controls", &stereoFunc->stereoNumDisparities,640, onNumDisparitiesSlide);
}

void mouseHandler(int event, int x, int y, int flags, void *param){

	switch(event){
	case CV_EVENT_LBUTTONDOWN:
	//l = cvGet2D(stereoFunc->depthM, x, y);
	printf("Distance to this object is: %f cm \n",(float)cvGet2D(stereoFunc->depthM, x, y).val[0]);
	break;
	}
}

void camera_gray(){
	grab->stereoGrabInitFrames();
	grab->stereGrabFrames();
	IplImage *frame1 = grab->imageLeft;
	IplImage *frame2 = grab->imageRight;
	IplImage *grayfrm2, *grayfrm1;
	CvMat *frem1;

	int height, width, step, channels, k, i_max;
	int *hist;

	//gambar kamera kiri
	uchar *datakiri;
	uchar **gray_arrkiri;

	//gambar kamera kanan
	uchar *datakanan;
	uchar **gray_arrkanan;

	//definisi gambar
	height = frame1->height; //tinggi gambar
	width = frame1->width;  //lebar gambar
	step = frame1->widthStep;	// array pada elementuntuk satu baris pada gambar
	channels = frame1->nChannels;	//channels R.G.B.
	datakiri = (uchar *)frame1->imageData;
	datakanan = (uchar *)frame2->imageData;

	//end definisi gambar

	char choice = 'z';
	while (choice != 'q'){

		grab->stereGrabFrames();
		frame1 = grab->imageLeft;
		frame2 = grab->imageRight;
		
		// untuk mengalokasikan blok memori dan mengembalikan pointer ke awal blok 
		gray_arrkiri = (uchar **)malloc(sizeof(uchar *)*(height + 1));
		// untuk mengalokasikan blok memori dan mengembalikan pointer ke awal blok 
		gray_arrkanan = (uchar **)malloc(sizeof(uchar *)*(height + 1));
		// Convert the RGB image to a grayscale image 
		for (int i = 0; i < height; i++){
			gray_arrkiri[i] = (uchar *)malloc(sizeof(uchar)*(width + 1));
			gray_arrkanan[i] = (uchar *)malloc(sizeof(uchar)*(width + 1));

			for (int j = 0; j < width; j++) {   //ISO grayscale image is 11%BLUE + 56% GREEN + 33% RED..... so converitng 1d array into 2d array   
				gray_arrkiri[i][j] = (0.11*datakiri[i*step + j*channels] + 0.56*datakiri[i*step + j*channels + 1] + 0.33*datakiri[i*step + j*channels + 2]);
				gray_arrkanan[i][j] = (0.11*datakanan[i*step + j*channels] + 0.56*datakanan[i*step + j*channels + 1] + 0.33*datakanan[i*step + j*channels + 2]);
				//histo_arr[i][j] = 0;
			}
		}

		grayfrm1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		grayfrm2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				grayfrm1->imageData[i*grayfrm1->widthStep + j*grayfrm1->nChannels] = gray_arrkiri[i][j]; // membuat gambar gray
				grayfrm2->imageData[i*grayfrm2->widthStep + j*grayfrm2->nChannels] = gray_arrkanan[i][j]; // membuat gambar gray
			}
		}
		cvShowImage("camera left", frame1);
		cvShowImage("camera right", frame2);
		cvShowImage("GrayScaled grayfrm1", grayfrm1);
		cvShowImage("GrayScaled grayfrm2", grayfrm2);
		if (cvWaitKey(15) == 27) break;
	}
	cvReleaseImage(&grayfrm1);
	cvReleaseImage(&grayfrm2);
	cvReleaseImage(&frame1);
	cvReleaseImage(&frame2);

	grab->stereoGrabStopCam();

}

void cameraMat(){
	grab->stereoGrabInitFrames();
	grab->stereGrabFrames();
	IplImage *frame1 = grab->imageLeft;
	IplImage *frame2 = grab->imageRight;

	Mat frmm1 = cvarrToMat(frame1);
	Mat frmm2 = cvarrToMat(frame2);

	while (1){
	grab->stereGrabFrames();

	imshow("cam2", frmm2);
	imshow("cam1", frmm1);

	if (waitKey(30) >= 0)
		break;
	}
}

void camera_grayv2(){
	grab->stereoGrabInitFrames();
	grab->stereGrabFrames();
	IplImage *frame1 = grab->imageLeft;
	IplImage *frame2 = grab->imageRight;
	IplImage *grayfrm2, *grayfrm1;
	CvMat *frem1;

	int height, width, step, channels, k, i_max;
	int *hist;

	int heightka, widthka, stepka, channelska;

	//gambar kamera kiri
	uchar *datakiri;
	uchar **gray_arrkiri;

	//gambar kamera kanan
	uchar *datakanan;
	uchar **gray_arrkanan;

	//definisi gambar
	height = frame1->height; //tinggi gambar
	width = frame1->width;  //lebar gambar
	step = frame1->widthStep;	// array pada elementuntuk satu baris pada gambar
	channels = frame1->nChannels;	//channels R.G.B.
	datakiri = (uchar *)frame1->imageData;
		
	heightka = frame2->height; //tinggi gambar
	widthka = frame2->width;  //lebar gambar
	stepka = frame2->widthStep;	// array pada elementuntuk satu baris pada gambar
	channelska = frame2->nChannels;	//channels R.G.B.
	datakanan = (uchar *)frame2->imageData;

	//end definisi gambar

	char choice = 'z';
	while (choice != 'q'){

		grab->stereGrabFrames();
		frame1 = grab->imageLeft;
		frame2 = grab->imageRight;

		// untuk mengalokasikan blok memori dan mengembalikan pointer ke awal blok 
		gray_arrkiri = (uchar **)malloc(sizeof(uchar *)*(height + 1));
		// untuk mengalokasikan blok memori dan mengembalikan pointer ke awal blok 
		gray_arrkanan = (uchar **)malloc(sizeof(uchar *)*(heightka + 1));
		// Convert the RGB image to a grayscale image 
		for (int i = 0; i < height; i++){
			gray_arrkiri[i] = (uchar *)malloc(sizeof(uchar)*(width + 1));

			for (int j = 0; j < width; j++) {   //ISO grayscale image is 11%BLUE + 56% GREEN + 33% RED..... so converitng 1d array into 2d array   
				gray_arrkiri[i][j] = (0.11*datakiri[i*step + j*channels] + 0.56*datakiri[i*step + j*channels + 1] + 0.33*datakiri[i*step + j*channels + 2]);
			}
		}

		for (int i = 0; i < heightka; i++){
			gray_arrkanan[i] = (uchar *)malloc(sizeof(uchar)*(widthka + 1));
			for (int j = 0; j < widthka; j++) {   //ISO grayscale image is 11%BLUE + 56% GREEN + 33% RED..... so converitng 1d array into 2d array   
				gray_arrkanan[i][j] = (0.11*datakanan[i*stepka + j*channelska] + 0.56*datakanan[i*step + j*channelska + 1] + 0.33*datakanan[i*step + j*channelska + 2]);
			}
		}

		grayfrm1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		grayfrm2 = cvCreateImage(cvSize(widthka, heightka), IPL_DEPTH_8U, 1);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				grayfrm1->imageData[i*grayfrm1->widthStep + j*grayfrm1->nChannels] = gray_arrkiri[i][j]; // membuat gambar gray
		//		grayfrm2->imageData[i*grayfrm2->widthStep + j*grayfrm2->nChannels] = gray_arrkanan[i][j]; // membuat gambar gray
			}
		}


		for (int i = 0; i < heightka; i++) {
			for (int j = 0; j < widthka; j++) {
		//		grayfrm1->imageData[i*grayfrm1->widthStep + j*grayfrm1->nChannels] = gray_arrkiri[i][j]; // membuat gambar gray
				grayfrm2->imageData[i*grayfrm2->widthStep + j*grayfrm2->nChannels] = gray_arrkanan[i][j]; // membuat gambar gray
			}
		}
		cvShowImage("camera left", frame1);
		cvShowImage("camera right", frame2);
		cvShowImage("GrayScaled grayfrm1", grayfrm1);
		cvShowImage("GrayScaled grayfrm2", grayfrm2);
		if (cvWaitKey(15) == 27) break;
	}
	cvReleaseImage(&grayfrm1);
	cvReleaseImage(&grayfrm2);
	cvReleaseImage(&frame1);
	cvReleaseImage(&frame2);

	grab->stereoGrabStopCam();

}

void grayscale(){
	IplImage * imag; IplImage * gray; IplImage * histogram;
	int height, width, step, channels, k, i_max;

	int *hist;
	uchar *data;
	uchar **gray_arr;
	uchar **histo_arr;

	imag = cvLoadImage("me.jpg");
	height = imag->height;
	width = imag->width;
	step = imag->widthStep;
	channels = imag->nChannels;
	data = (uchar *)imag->imageData;


	while (1){
	hist = (int *)calloc(256, sizeof(int));
	histo_arr = (uchar **)malloc(sizeof(uchar *)*(height + 1));
	gray_arr = (uchar **)malloc(sizeof(uchar *)*(height + 1));
	
			// Convert the RGB image to a grayscale image 
			for (int i = 0; i < height; i++){
				histo_arr[i] = (uchar *)malloc(sizeof(uchar)*(width + 1));
				gray_arr[i] = (uchar *)malloc(sizeof(uchar)*(width + 1));

				for (int j = 0; j < width; j++) {   // we know that ISO grayscale image is 11%BLUE + 56% GREEN + 33% RED..... so converitng 1d array into 2d array   
					gray_arr[i][j] = (0.11*data[i*step + j*channels] + 0.56*data[i*step + j*channels + 1] + 0.33*data[i*step + j*channels + 2]);
					histo_arr[i][j] = 0;
				}
			}

			//FILE * pFile; pFile = fopen("matriks-histogram.txt", "w");
			////Construct the histogram array 
			//for(int i=0;i<height;i++){  
			//	fprintf(pFile,"baris %d --> ",(i+1));     //
			//	for (int j=0;j<width;j++) {   //gray_arr[i][j]+=80;   
			//		k=gray_arr[i][j];   
			//		fprintf (pFile, "| %d | ", gray_arr[i][j]);   
			//		hist[k]++;  
			//	}  
			//	fprintf(pFile,"\n"); 
			//} 
			//fclose (pFile); 
			//system("notepad.exe matriks-histogram.txt");

			gray = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					gray->imageData[i*gray->widthStep + j*gray->nChannels] = gray_arr[i][j];
				}
			}

			cvNamedWindow("Original Image", CV_WINDOW_NORMAL); cvMoveWindow("Original Image", 100, 100); cvShowImage("Original Image", imag);
			cvNamedWindow("GrayScaled Image", CV_WINDOW_NORMAL); cvMoveWindow("GrayScaled Image", 500, 100); cvShowImage("GrayScaled Image", gray);
			cvWaitKey(0);
		}

	cvReleaseImage(&imag); cvReleaseImage(&gray); cvReleaseImage(&histogram);
	free(gray_arr); free(histo_arr);
}

void capturevid(){
	grab->stereoGrabInitFrames();
	grab->stereGrabFrames();
	IplImage *frame1 = grab->imageLeft;
	IplImage *frame2 = grab->imageRight;
	
	Mat matImg(frame1);
	Mat matImg2(frame2);

	Size frame_size(320, 240);
	int frames_per_second = 50;

	//Create and initialize the VideoWriter object 
	VideoWriter oVideoWriter("MyVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), frames_per_second, frame_size, true);

	stereoFunc->stereoInit(grab);

	while(1){
		grab->stereGrabFrames();
		//write the video frame to the file
		oVideoWriter.write(matImg);
		imshow("display",matImg);
		imshow("display1", matImg2);
		if (cvWaitKey(15) == 27) break;
	}
	//Flush and close the video file
	oVideoWriter.release();
}

int capturevideo(){
	//Open the default video camera
	VideoCapture cap(2);

	// if not success, exit program
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video camera" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	int frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH)); //get the width of frames of the video
	int frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT)); //get the height of frames of the video

	Size frame_size(frame_width, frame_height);
	int frames_per_second = 15;

	//Create and initialize the VideoWriter object 
	VideoWriter oVideoWriter("MyVideo.avi",CV_FOURCC('M','J','P','G'),frames_per_second, frame_size, true);

	//If the VideoWriter object is not initialized successfully, exit the program
	if (oVideoWriter.isOpened() == false)
	{
		cout << "Cannot save the video to a file" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	string window_name = "My Camera Feed";
	namedWindow(window_name); //create a window called "My Camera Feed"

	while (true)
	{
		Mat frame;
		bool isSuccess = cap.read(frame); // read a new frame from the video camera

		//Breaking the while loop if frames cannot be read from the camera
		if (isSuccess == false)
		{
			cout << "Video camera is disconnected" << endl;
			cin.get(); //Wait for any key press
			break;
		}

		/*
		Make changes to the frame as necessary
		e.g.
		1. Change brightness/contrast of the image
		2. Smooth/Blur image
		3. Crop the image
		4. Rotate the image
		5. Draw shapes on the image
		*/

		//write the video frame to the file
		oVideoWriter.write(frame);

		//show the frame in the created window
		imshow(window_name_detect, frame);

		//Wait for for 10 milliseconds until any key is pressed.  
		//If the 'Esc' key is pressed, break the while loop.
		//If any other key is pressed, continue the loop 
		//If any key is not pressed within 10 milliseconds, continue the loop 
		if (waitKey(10) == 27)
		{
			cout << "Esc key is pressed by the user. Stopping the video" << endl;
			break;
		}
	}

	//Flush and close the video file
	oVideoWriter.release();

}

void histogram(){
		Mat image = imread("me.jpg", CV_LOAD_IMAGE_UNCHANGED);
		Mat imgray = imread("me.jpg", CV_LOAD_IMAGE_UNCHANGED);

		int valcolour[3];

		//rgb
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				valcolour[0] = image.at<Vec3b>(i, j)[0]; //b
				valcolour[1] = image.at<Vec3b>(i, j)[1]; //g
				valcolour[2] = image.at<Vec3b>(i, j)[2]; //r

				if (image.at<Vec3b>(i, j)[0]<100 && image.at<Vec3b>(i, j)[1]<100 && image.at<Vec3b>(i, j)[2]<100){
					valcolour[0] = image.at<Vec3b>(i, j)[0]=0; //b
					valcolour[1] = image.at<Vec3b>(i, j)[1]=0; //g
					valcolour[2] = image.at<Vec3b>(i, j)[2]=0; //r
				}
				printf("%d %d %d \n", valcolour[0], valcolour[1], valcolour[2]);
				// do something with BGR values...
			}
		}
		//gray

		cvtColor(imgray, imgray,CV_BGR2GRAY);
		int gray;
		for (int j = 0; j<imgray.rows; j++)
		{
			for (int i = 0; i<imgray.cols; i++)
			{
				if (imgray.at<uchar>(j,i) < 100)
				{
					imgray.at<uchar>(j, i) = 0;
				}
				printf(" %d",imgray.at<uchar>(j, i));

			}
		}

		imshow("gray", imgray);
		imshow("colour", image);

		waitKey(0);

}

void detectAndDisplay(Mat frame){
	Mat frame_gray;
	int red, green, blue;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, gray);
	//imshow("frame_gray", frame_gray);
	//imshow("Equalize", gray);

	//-- Detect waste
	waste_cascade.detectMultiScale(frame_gray, waste, 1.1, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(24, 24));

	for (size_t i = 0; i < waste.size(); i++)
	{

		Point center(waste[i].x + waste[i].width / 2, waste[i].y + waste[i].height / 2);
		Point p_atas(waste[i].x, waste[i].y);
		Point p_bawah(waste[i].x + waste[i].width, waste[i].y + waste[i].height);
		roi_top = p_atas; roi_bot = p_bawah;

		Point center1(waste[i].x, waste[i].y);

		rectangle(frame, p_atas, p_bawah, Scalar(0, 255, 0), 2);
		//ellipse(frame, center, Size(3, 3), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);
		//ellipse(frame, center, Size(waste[i].width / 2, waste[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 1, 8, 0);
		/*Vec3b intensity = frame.at<Vec3b>((waste[i].y + waste[i].height / 2), (waste[i].x + waste[i].width / 2));
		blue = intensity.val[0]; green = intensity.val[1]; red = intensity.val[2];
		String r = to_string(red);  String g = to_string(green); String b = to_string(blue);
		String rgb = "R:" + r + "  G:" + g + "  B:" + b;
		//putText(frame, rgb, p_atas, FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 255), 1);*/

	}
	//-- Show what you got
	//imshow(window_name, frame);
	//printf("R: %d | G: %d | B: %d \n", red, green, blue);*/
}

void tracking() {

	double precTick = ticks;
	ticks = (double)getTickCount();

	double dT = (ticks - precTick) / getTickFrequency(); //seconds

	//cout << "Start Tracking..." << endl;
	setIdentity(kf.transitionMatrix);

	// Measure Matrix H
	// [ 1 0 0 0 0 0 ]
	// [ 0 1 0 0 0 0 ]
	// [ 0 0 0 0 1 0 ]
	// [ 0 0 0 0 0 1 ]
	kf.measurementMatrix = Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(7) = 1.0f;
	kf.measurementMatrix.at<float>(16) = 1.0f;
	kf.measurementMatrix.at<float>(23) = 1.0f;

	// Process Noise Covariance Matrix Q
	// [ Ex   0   0     0     0    0  ]
	// [ 0    Ey  0     0     0    0  ]
	// [ 0    0   Ev_x  0     0    0  ]
	// [ 0    0   0     Ev_y  0    0  ]
	// [ 0    0   0     0     Ew   0  ]
	// [ 0    0   0     0     0    Eh ]
	//setIdentity(kf.processNoiseCov, Scalar(1e-2));
	kf.processNoiseCov.at<float>(0) = 1e-2;
	kf.processNoiseCov.at<float>(7) = 1e-2;
	kf.processNoiseCov.at<float>(14) = 5.0f;
	kf.processNoiseCov.at<float>(21) = 5.0f;
	kf.processNoiseCov.at<float>(28) = 1e-2;
	kf.processNoiseCov.at<float>(35) = 1e-2;

	// Measures Noise Covariance Matrix R
	setIdentity(kf.measurementNoiseCov, Scalar(1e-1));

	if (found)
	{
		//cout << "found" << endl;

		// >>>> Matrix A
		kf.transitionMatrix.at<float>(2) = dT;
		kf.transitionMatrix.at<float>(9) = dT;
		// <<<< Matrix A

		//cout << "dT:" << dT << endl;

		state = kf.predict();
		//cout << "State post:" << endl << state << endl;

		Rect predRect;
		predRect.width = state.at<float>(4);
		predRect.height = state.at<float>(5);
		predRect.x = state.at<float>(0) - predRect.width / 2;
		predRect.y = state.at<float>(1) - predRect.height / 2;

		track_roi_top = Point(predRect.x, predRect.y);
		track_roi_bot = Point(predRect.x + predRect.width, predRect.y + predRect.height);

		if (track_roi_bot.x >= 320) track_roi_bot.x = 320;
		if (track_roi_bot.y >= 240) track_roi_bot.y = 240;
		if (track_roi_top.x <= 0) track_roi_top.x = 0;
		if (track_roi_top.y <= 0) track_roi_top.y = 0;

		Point center;
		center.x = state.at<float>(0);
		center.y = state.at<float>(1);
		//circle(frame_cpy, center, 2, CV_RGB(255, 0, 0), -1);
		rectangle(frame_cpy, predRect, CV_RGB(255, 0, 0), 2);

		//PointCenter(center);
		//ellipse(frame_cpy, center, Size(3, 3), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);

		/*stringstream sstr;
		sstr << "(" << center.x << "," << center.y << ")";
		putText(frame_cpy, sstr.str(), Point(center.x + 3, center.y - 3), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 20, 20), 2);*/
	}

	// >>>>> Detection result
	for (size_t i = 0; i < waste.size(); i++)
	{

		rectangle(res, waste[i], CV_RGB(0, 255, 0), 2);

		Point center;
		center.x = waste[i].x + waste[i].width / 2;
		center.y = waste[i].y + waste[i].height / 2;
		circle(res, center, 2, CV_RGB(20, 150, 20), -1);

		stringstream sstr;
		sstr << "(" << center.x << "," << center.y << ")";
		//putText(frame_cpy, sstr.str(), Point(center.x + 3, center.y - 3), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20, 150, 20), 2);
	}

	// >>>>> Kalman Update
	if (waste.size() == 0)
	{
		notFoundCount++;
		//cout << "notFoundCount:" << notFoundCount << endl;
		if (notFoundCount >= 10)
		{
			found = false;
		}
		/*else
		kf.statePost = state;*/
	}
	else
	{
		notFoundCount = 0;

		meas.at<float>(0) = waste[0].x + waste[0].width / 2;
		meas.at<float>(1) = waste[0].y + waste[0].height / 2;
		meas.at<float>(2) = (float)waste[0].width;
		meas.at<float>(3) = (float)waste[0].height;

		if (!found) // First detection!
		{
			// >>>> Initialization
			kf.errorCovPre.at<float>(0) = 1; // px
			kf.errorCovPre.at<float>(7) = 1; // px
			kf.errorCovPre.at<float>(14) = 1;
			kf.errorCovPre.at<float>(21) = 1;
			kf.errorCovPre.at<float>(28) = 1; // px
			kf.errorCovPre.at<float>(35) = 1; // px

			state.at<float>(0) = meas.at<float>(0);
			state.at<float>(1) = meas.at<float>(1);
			state.at<float>(2) = 0;
			state.at<float>(3) = 0;
			state.at<float>(4) = meas.at<float>(2);
			state.at<float>(5) = meas.at<float>(3);
			// <<<< Initialization

			kf.statePost = state;

			found = true;
		}
		else
			//cout << "Kalman Correction" << endl;
			kf.correct(meas); // Kalman Correction

	}
	// <<<<< Kalman Update

	// Final result
	//imshow("Tracking", res);
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

double ecu_top=0.0;
double ecu_bot=0.0;
float distance_obj;
int f_detect = 10;

void enchanceTracking() {
	int val = 0; int reloop = 0; int c_line = 3;
	detect_roi_top = 0; detect_roi_bot = 0;
	detect_roi_left = 0; detect_roi_right = 0;
	int mines = 0;
	Point last_point_top = track_roi_top; 
	Point last_point_bot = track_roi_bot;
	
	/// track_roi original
	while (reloop <= 50) {
		cout << "jalan loop" << endl;
		for (int i = track_roi_top.x; i < track_roi_bot.x; i++) {
			val = enchanceImage.at<uchar>(track_roi_top.y, i);
			if (val >= thresholdC - mines) detect_roi_top++;
		}
		for (int i = track_roi_top.x; i < track_roi_bot.x; i++) {
			val = enchanceImage.at<uchar>(track_roi_bot.y, i);
			if (val >= thresholdC - mines) detect_roi_bot++;
		}
		for (int i = track_roi_top.y; i < track_roi_bot.y; i++) {
			val = enchanceImage.at<uchar>(i, track_roi_top.x);
			if (val >= thresholdC - mines) detect_roi_left++;
		}
		for (int i = track_roi_top.y; i < track_roi_bot.y; i++) {
			val = enchanceImage.at<uchar>(i, track_roi_bot.x);
			if (val >= thresholdC - mines) detect_roi_right++;
		}
		//cout << endl << detect_roi_top << " " << detect_roi_right << " " << detect_roi_bot << " " << detect_roi_left << endl;

		/// Check for conditioning
		/// Adjust Smaller
		cout << "jalan loop2" << endl;
		if (detect_roi_top == 0 && detect_roi_bot == 0) {
			track_roi_top.y += 1;
			track_roi_bot.y += -1;
		}
		if (detect_roi_left == 0 && detect_roi_right == 0 ) {
			track_roi_top.x += 1;
			track_roi_bot.x += -1;
		}
		if (detect_roi_bot >= c_line && detect_roi_top == 0) {
			track_roi_top.y += 1;
		}
		if (detect_roi_top >= c_line && detect_roi_bot == 0) {
			track_roi_bot.y += -1;
		}
		if (detect_roi_right >= c_line && detect_roi_left == 0) {
			track_roi_top.x += 1;
		}
		if (detect_roi_left >= c_line && detect_roi_right == 0) {
			track_roi_bot.x += -1;
		}

		/// Adjust bigger
		cout << "jalan loop3" << endl;
		if (detect_roi_top >= c_line && detect_roi_bot >= c_line) {
			track_roi_top.y += -1; 
			track_roi_bot.y += 1; 
		}
		if (detect_roi_left >= c_line && detect_roi_right >= c_line) {
			track_roi_top.x += -1; 
			track_roi_bot.x += 1;
		}
		if (detect_roi_bot >= c_line && detect_roi_top < c_line) {
			track_roi_bot.y += 1;
			track_roi_top.y += 1;
		}
		if (detect_roi_top >= c_line && detect_roi_bot < c_line) {
			track_roi_bot.y += -1;
			track_roi_top.y += -1;
		}
		if (detect_roi_right >= c_line && detect_roi_left < c_line) {
			track_roi_bot.x += 1;
			track_roi_top.x += 1;
		}
		if (detect_roi_left >= c_line && detect_roi_right < c_line) {
			track_roi_bot.x += -1;
			track_roi_top.x += -1;
		}

		cout << "jalan loop4" << endl;
		if (track_roi_bot.x >= 320) track_roi_bot.x = 320;
		if (track_roi_bot.y >= 240) track_roi_bot.y = 240;
		if (track_roi_top.x <= 0) track_roi_top.x = 0;
		if (track_roi_top.y <= 0) track_roi_top.y = 0;

		reloop++;
		//cout << endl << "thrs: " << thresholdC << "   reloop: " << reloop << endl;
		cout << "jalan loop5" << endl;
		///cout << endl << "stop enchanceTracking" << endl;
		if (detect_roi_top <= 2 && detect_roi_bot <= 2 && detect_roi_left <= 2 && detect_roi_right <= 2) {
			break;
		}

	}
	cout << "jalan loop6" << endl;
	int trc_width, trc_height;
	trc_width = track_roi_bot.x - track_roi_top.x;
	trc_height = track_roi_bot.y - track_roi_top.y;
	roi_center.x = track_roi_top.x + (trc_width / 2);
	roi_center.y = track_roi_top.y + (trc_height / 2);
	PointCenter(roi_center);

	cout << "jalan loop7" << endl;
	
	meas_distance(meas_dist);
	cout << "loop8" << endl;
	distance_obj = 0.0;
	cout << "loop9" << endl;
	for (auto i = meas_dist.begin(); i != meas_dist.end(); ++i) distance_obj = *i;
	distance_obj = (distance_obj * 100) / 100;
	meas_dist.clear();
	cout << "loop10" << endl;
	circle(frame_cpy, roi_center, 2, CV_RGB(255, 0, 0), -1);
	rectangle(enchanceImage, track_roi_top, track_roi_bot, Scalar(255, 255, 255), 2);
	rectangle(frame_cpy, track_roi_top, track_roi_bot, Scalar(255, 0, 0), 2);

	ecu_top = norm(last_point_top - track_roi_top);
	ecu_bot = norm(last_point_bot - track_roi_bot);
	//cout << endl << "e_top: " << ecu_top << " ecu_bot: " << ecu_bot << endl;
	cout << "buyar enchance" << endl;
}

void print_distance_angle() {
	if (ecu_top > f_detect || ecu_top > f_detect) flag_class_result = 0;
	else {
		flag_class_result = 1;
		Point line_center(160, roi_center.x);
		double line_a = 240 - roi_center.y;
		double line_b = roi_center.x - 160;
		float meas_degre = atan2(line_a, line_b);
		meas_degre = meas_degre * 180 / 3.14;
		if (meas_degre > 90) meas_degre = (meas_degre - 90)*-1;
		else meas_degre = 90 - meas_degre;

		/*stringstream sstr;
		sstr << "(" << distance_obj << " : " << meas_degre << ")";*/
		sprintf(buffer, "(%.2f : %.2f)", distance_obj, meas_degre);
		if(flag_bukan_sampah==1)
			putText(ClassImg, buffer, Point(roi_center.x, roi_center.y), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 20, 20), 2);
	}
}

void thresh_callback(int, void*) {
	Canny(enchanceImage, enchanceImage, thresholdC, thresholdC * 2, 3);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "detection");
    ros::NodeHandle nh;

    //view_camera();		

	if(CALIBRATION) playcal();
	loadcorrelation();
	grab->stereoGrabInitFrames();
	grab->stereGrabFrames();
	stereoFunc->stereoInit(grab);

	time_t start, end;
	int counter = 0;
	double sec;
	double fps;

	/// Load HOG
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
	char choice='z';
	while(choice!='q'){
		

		// fps counter begin
		if (counter == 0){
			time(&start);
		}
		cout << "mulai grab" << endl;
		//stereoCorrelationControl();
		grab->stereGrabFrames();
		cout << "after grab" << endl;
		stereoFunc->stereoCorrelation(grab);
		cout << "After Corelasi" << endl;
		if (cvWaitKey(1) == 27) {
			
			cout << "Capturing ROI..." << endl;
			//sprintf(buffer1, "C:\\Users\\irfan\\OneDrive\\Documents\\TA\\TA_Akhir\\Project\\DataTraining\\img%u.png", imcount++);
			//sprintf(buffer1, "/home/kisron/catkin_workspace/Project_Bang_Salimi/DataTraining/DataTrainAll/img%u.png", imcount++);
			sprintf(buffer1, "/home/kisron/catkin_workspace/Data_Training/img%u.png", imcount++);
			imwrite(buffer1, crop_image);
			imshow("Cropped image", crop_image);
		}

		if (cvWaitKey(1) == 32) {
			flag_print = 1;
		}
		//Mat read_cam_left = imread("C:\\Users\\irfan\\OneDrive\\Documents\\TA\\TA_Akhir\\Project\\bismillah vision 8 sukses\\bismillah vision\\cam_left.jpg");
		Mat read_cam_left = imread("/home/kisron/catkin_workspace/images/cam_left.jpg");
		frame_cpy = read_cam_left.clone();
		enchanceImage = read_cam_left.clone();
		cv::cvtColor(enchanceImage, enchanceImage, CV_BGR2GRAY);
		cv::blur(enchanceImage, enchanceImage, Size(3, 3));
		pre_crop_image = frame_cpy.clone();
		res = frame_cpy.clone();
		
		cv::createTrackbar(" Canny thresh:", "Canny", &thresholdC, 255, thresh_callback);
		thresh_callback(0, 0);
		cout << "before extract" << endl;
		if (!frame_cpy.empty()) {
			if(run==0 || run%5==0) 
			detectAndDisplay(frame_cpy);
			tracking(); run++;
			//enchanceTracking();
			imshow("Canny", enchanceImage);
			if (run == 10000) run = 0;
			ClassImg = frame_cpy.clone();

			if (track_roi_top.x <= 0 || track_roi_top.y <= 0 || track_roi_bot.x <= 0 || track_roi_bot.y <= 0) {
				Rect ROI(roi_top, roi_bot);
				crop_image = pre_crop_image(ROI);
			}
			else {
				Rect ROI(track_roi_top, track_roi_bot);
				crop_image = pre_crop_image(ROI);
			}
			
			imshow("crop_image", crop_image);
			
			ExtractFeature(pre_crop_image, 0); /// Feature Extraction
			reduceFeatureUsingPCAinSVM(eMat, reduced, vectorHogGlcm, false); /// Reduce HOG using PCA
			for (auto i = vec_glcm.begin(); i != vec_glcm.end(); ++i) vectorHogGlcm.push_back(*i);
			int no = 0;
			for (auto i = vectorHogGlcm.begin(); i != vectorHogGlcm.end(); ++i) imgToSvm.at<float>(0, no++) = *i;
			vec_glcm.clear(); vectorHogGlcm.clear(); /// Empty the vector
			Classifier();
			print_distance_angle();
		}

		time(&end);
		counter++;
		sec = difftime(end, start);

		fps = counter / sec;
		if (sec<60){
			if(flag_print==1) cout << "counter: " << counter << " sec: " << sec << " ";
		}

		/// Print fps
		String s_fps;
		Point p_fps(5, 15);
		s_fps = to_string(fps);
		s_fps = "fps" + s_fps;
		putText(ClassImg, s_fps, p_fps, FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 255), 2);

		/// Print frame counter
		Point p_fcount(260, 15);
		s_fps = to_string(f_counter++);
		s_fps = "#" + s_fps;
		putText(ClassImg, s_fps, p_fcount, FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 255), 2);

		/// overflow protection
		if (counter == (INT_MAX - 1000))
			counter = 0;
		///fps counter end

		imshow("Classification", ClassImg);

	}
	//Flush and close the video file
//	oVideoWriter.release();
	destroyAllWindows();
	grab->stereoGrabStopCam();

	return 0;

}
