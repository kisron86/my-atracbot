#include <ros/ros.h>
#include "my_roscpp_library/my_super_roscpp_library.h"
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/ml.hpp"

#include <time.h> // to calculate time needed
#include <limits.h> // to get INT_MAX, to protect against overflow

#include <iomanip>
#include <iostream>
#include <stdlib.h>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "my_roscpp_library/my_stereofunction.h"
#include "my_roscpp_library/my_stereograb.h"
#include "my_roscpp_library/my_glcm.h"
#include "my_roscpp_library/my_hog.h"

#define CALIBRATION 0

using namespace std;
using namespace cv;

StereoGrab* grab= new StereoGrab();
StereoFunction* stereoFunc = new StereoFunction();


//#define M_PI 3.1415

//////////////////////////------------CAPTURE IMAGE-------------/////////////////////////////////////////////////////////////
void capture(){

	VideoCapture capr(2), capl(1);
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
			if(framel.empty() || framer.empty()) break;
				imshow("Left", framel);
				imshow("Right", framer);
			if(choice == 'c') {
		//save files at proper locations if user presses 'c'
				stringstream l_name, r_name;
				l_name << "left" << setw(2) << setfill('0') << count << ".jpg";
				r_name << "right" << setw(2) << setfill('0') << count << ".jpg";
				imwrite(l_name.str(), cvarrToMat(grab->imageLeft));
				imwrite(r_name.str(), cvarrToMat(grab->imageRight));
				cout << "Saved set " << count << endl;
				count++;
			}
		choice = char(waitKey(1));
		}
	capl.release();
	capr.release();
}

void view_camera(){
	int count = 0;

	grab->stereoGrabInitFrames();
	grab->stereGrabFrames();
	IplImage *frame1 = grab->imageLeft;
	IplImage *frame2 = grab->imageRight;

	char choice = 'z';
	while(choice!='q'){
		
		grab->stereGrabFrames();
		frame1 = grab->imageLeft;
		cvShowImage("camera left", frame1);
		frame2 = grab->imageRight;
		cvShowImage("camera right", frame2);
		if (choice == 'z') {
			//save files at proper locations if user presses 'c'
			stringstream l_name, r_name;
			l_name << "left" << setw(2) << setfill('0') << count << ".jpg";
			r_name << "right" << setw(2) << setfill('0') << count << ".jpg";
			imwrite(l_name.str(), cvarrToMat(grab->imageLeft));
			//imwrite(r_name.str(), cvarrToMat(grab->imageRight));
			cout << "Saved set " << count << endl;
			count++;
		}
		if(cvWaitKey(15)==27) break;
	}
}

void playcal(){
	stereoFunc->stereoCalibration(grab);
}

void loadcorrelation(){
		ifstream params;
	params.open("/home/kisron/catkin_workspace/Correlation.txt");
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
		stereoFunc->stereoPreFilterSize = 99;
		stereoFunc->stereoPreFilterCap = 56;//32;//63; 
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

void onTextureBarSlide(int pos)
{
	stereoFunc->stereoDispTextureThreshold = cvGetTrackbarPos("Texture th", "Stereo Controls");
	if(stereoFunc->stereoDispTextureThreshold) 
		stereoFunc->stereoCorrelation(grab);
}

void onUniquenessBarSlide(int pos)
{
	stereoFunc->stereoDispUniquenessRatio = cvGetTrackbarPos("Uniqueness", "Stereo Controls");
	if(stereoFunc->stereoDispUniquenessRatio>=0)
		stereoFunc->stereoCorrelation(grab);
}

void onNumDisparitiesSlide(int pos)
{
	stereoFunc->stereoNumDisparities = cvGetTrackbarPos("Num.Disp", "Stereo Controls");
	while(stereoFunc->stereoNumDisparities%16!=0 || stereoFunc->stereoNumDisparities==0)
		stereoFunc->stereoNumDisparities++;

	stereoFunc->stereoCorrelation(grab);
}

void onPreFilterSizeBarSlide(int pos)
{
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

void onPreFilterCapBarSlide(int pos)
{
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

	int height, width, step, channels, k, i_max;

	//gambar kamera kiri
	uchar *datakiri;
	uchar **gray_arrkiri;


	Mat frmm1 = cvarrToMat(frame1);
	Mat frmm2 = cvarrToMat(frame2);



	//definisi gambar
	height = frame1->height; //tinggi gambar
	width = frame1->width;  //lebar gambar
	step = frame1->widthStep;	// array pada elementuntuk satu baris pada gambar
	channels = frame1->nChannels;	//channels R.G.B.
	datakiri = (uchar *)frame1->imageData;

	while (1){
	grab->stereGrabFrames();
	frame1 = grab->imageLeft;
	frame2 = grab->imageRight;

	// untuk mengalokasikan blok memori dan mengembalikan pointer ke awal blok 
	gray_arrkiri = (uchar **)malloc(sizeof(uchar *)*(height + 1));
	

	// Convert the RGB image to a grayscale image 
	for (int i = 0; i < height; i++){
		gray_arrkiri[i] = (uchar *)malloc(sizeof(uchar)*(width + 1));

		for (int j = 0; j < width; j++) {   //ISO grayscale image is 11%BLUE + 56% GREEN + 33% RED..... so converitng 1d array into 2d array   
			gray_arrkiri[i][j] = (0.11*datakiri[i*step + j*channels] + 0.56*datakiri[i*step + j*channels + 1] + 0.33*datakiri[i*step + j*channels + 2]);
		}
	}
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
	int frames_per_second = 10;

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
	VideoCapture cap(0);

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
		imshow(window_name, frame);

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

void savevideo(){

	Size frame_size(320, 240);
	int frames_per_second = 10;

	//Create and initialize the VideoWriter object 
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "target_detection");
    ros::NodeHandle nh;

    	//capture();
	//HOG();
	//view_camera();
	//histogram();
	
	//capturevid();
	//capturevideo();
	//cameraMat();
	//camera_grayv2();
	//camera_gray();
	// grayscale();
	//MENGHITUNG FPS
	
	// fps counter begin
	time_t start, end;
	int counter = 0;
	double sec;
	double fps;
	// fps counter end
	
	//if(CALIBRATION) playcal();
	//loadcorrelation();
	grab->stereoGrabInitFrames();
	grab->stereGrabFrames();
	stereoFunc->stereoInit(grab);

	int count = 0;
	char choice='z';
	int itung=0;
	while(choice!='q'){
		
		// fps counter begin
		/*if (counter == 0){
			time(&start);
		}*/
		// fps counter end

		//stereoCorrelationControl();
		stereoFunc->stereoCorrelation(grab);
		//if (itung<2000){
		////menyimpan file tekan 'c'
		//stringstream depth;
		//depth << "salahhhhh" << setw(2) << setfill('0') << count << ".jpg";
		//imwrite(depth.str(), cvarrToMat(stereoFunc->vdisp));
		//cout << "Saved set " << count << endl;
		//count++;
		//itung++;
		//waitKey(10);
		//}

		if (choice == 'c') {
			//menyimpan file tekan 'c'
			stringstream depth;
			depth << "BBkiriori" << setw(2) << setfill('0') << count << ".jpg";
			imwrite(depth.str(), cvarrToMat(grab->imageLeft));
			depth << "BBdepth" << setw(2) << setfill('0') << count << ".jpg";
			imwrite(depth.str(), cvarrToMat(stereoFunc->vdisp));
			cout << "Saved set " << count << endl;
			count++;
		}
		grab->stereGrabFrames();
		choice = char(waitKey(1));


		//// fps counter begin
		//time(&end);
		//counter++;
		//sec = difftime(end, start);
		//fps = counter / sec;
		//if (counter > 30)
		//	printf("%.2f\n", fps);
		//// overflow protection
		//if (counter == (INT_MAX - 1000))
		//	counter = 0;
		//// fps counter end


		if(cvWaitKey(10) == 27) break;
	}
	//Flush and close the video file
	/*oVideoWriter.release();*/
	destroyAllWindows();
	grab->stereoGrabStopCam();

	return 0;	

    
}
