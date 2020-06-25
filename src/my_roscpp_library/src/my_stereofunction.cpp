#include <ros/ros.h>
#include <my_roscpp_library/my_stereofunction.h>
#include <my_roscpp_library/my_stereograb.h>
#include <opencv2/opencv.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h> //cxtypes.h
#include <stdio.h>
#include <iomanip>
#include "math.h" 



#if defined(_MSC_VER)
#include <tchar.h>
#include <strsafe.h>
#include <windows.h>
#pragma comment(lib, "Ws2_32.lib")
#elif defined(__GNUC__) || defined(__GNUG__)
#include <dirent.h>
#endif

string window_name = "Deteksi Manusia";
//VideoCapture cap(0);

std::string cascadeName = "bismillah20.xml";
cv::CascadeClassifier cascade;
int scale = 1;
int i;
float meas_dist = 0.0;

#define CETAK 1

int fileNO = 0;
IplImage *r_detect, *g_detect, *b_detect, *r_detect_r, *g_detect_r, *b_detect_r ;
int threshold, blobArea;
CvFont font;
int col = 0;
int column[320];
cv::Point p_center;

using namespace cv;

void StereoFunction::stereoInit(StereoGrab* grabb)
{
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 1.4f, CV_AA);
	
	_M1 = (CvMat *)cvLoad("CalibFile/M1.yml");
	_M2 = (CvMat *)cvLoad("CalibFile/M2.yml");
	_T  = (CvMat *)cvLoad("CalibFile/T.yml");
	mx1 = (CvMat *)cvLoad("CalibFile/mx1.yml");
	my1 = (CvMat *)cvLoad("CalibFile/my1.yml");
	mx2 = (CvMat *)cvLoad("CalibFile/mx2.yml");
	my2 = (CvMat *)cvLoad("CalibFile/my2.yml");
	//_Q = (CvMat *)cvLoad("CalibFile/Q.yml");
	_CamData = (CvMat *)cvLoad("CalibFile/CamData.yml");

	//READ In FOCAL LENGTH, SENSOR ELEMENT SIZE, XFOV, YFOV
	//0: fx(pixel), 1: fy(pixel), 2: B (baseline), 3: f(mm), 4: sensor element size, 5: baseline in mm
		/*reprojectionVars[0] = cvmGet(_M1,0,0);
		reprojectionVars[1] = cvmGet(_M1,0,0);
		reprojectionVars[2] = (-1)*cvmGet(_T,0,0);
		reprojectionVars[3] = cvmGet(_CamData, 0, 0);
		reprojectionVars[4] = cvmGet(_CamData, 0, 1);
		reprojectionVars[5] = cvmGet(_CamData, 0, 2);*/


		//Loading images
		img1 = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 1);		
		img2 = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 1);
		imageSize = cvSize(img1 -> width,img1 ->height);
		
		img1r = cvCreateMat( imageSize.height,imageSize.width, CV_8U );		//rectified left image
		img2r = cvCreateMat( imageSize.height,imageSize.width, CV_8U );		//rectified right image
		disp  = cvCreateMat( imageSize.height,imageSize.width, CV_16S );	//disparity map
		vdisp = cvCreateMat( imageSize.height,imageSize.width, CV_8U );
		depthM = cvCreateMat(imageSize.height, imageSize.width, CV_32F);
				
		
		thres_img = cvCreateImage( imageSize, img1->depth, 1);
		blobs_img = cvCreateImage( imageSize, img1->depth, 3);
		
		img_detect = cvCreateImage(imageSize, IPL_DEPTH_8U, 3);
		r_detect = cvCreateImage(imageSize,8,1);//subpixel
		r_detect_r = cvCreateImage(imageSize,8,1);
		g_detect = cvCreateImage(imageSize,8,1);//subpixel
		g_detect_r = cvCreateImage(imageSize,8,1);
		b_detect = cvCreateImage(imageSize,8,1);//subpixel
		b_detect_r = cvCreateImage(imageSize,8,1);
		
		pair = cvCreateMat( imageSize.height, imageSize.width*2,CV_8UC3 ); 
}

void StereoFunction::stereoCalibration(StereoGrab* grabb){

	int  nx=11, ny=7, frame = 0, n_boards =30, N;
	int count1 = 0,count2 = 0, result1=0, result2=0;	
    int  successes1 = 0,successes2 = 0;
   	const int maxScale = 1;
	const float squareSize = 2.0f;		//Set this to your actual square size
	CvSize imageSize = {0,0};
	CvSize board_sz = cvSize( nx,ny );

	int i, j, n = nx*ny, N1 = 0, N2 = 0;
	
    vector<CvPoint2D32f> points[2];
	vector<int> npoints;
	vector<CvPoint3D32f> objectPoints;
	vector<CvPoint2D32f> temp1(n); 
	vector<CvPoint2D32f> temp2(n);
    
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
	double Q[4][4];
	CvMat _Qcalib  = cvMat(4, 4, CV_64F, Q);
    CvMat _M1calib = cvMat(3, 3, CV_64F, M1 );
    CvMat _M2calib = cvMat(3, 3, CV_64F, M2 );
    CvMat _D1 	   = cvMat(1, 5, CV_64F, D1 );
    CvMat _D2      = cvMat(1, 5, CV_64F, D2 );
    CvMat _R       = cvMat(3, 3, CV_64F, R );
    CvMat _Tcalib  = cvMat(3, 1, CV_64F, T );
    CvMat _E       = cvMat(3, 3, CV_64F, E );
    CvMat _F       = cvMat(3, 3, CV_64F, F );
	
	//Start webcam
		printf("\nWebcams are starting ...\n");
		grabb->stereoGrabInitFrames();
		grabb->stereGrabFrames();
		IplImage *frame1 = grabb->imageLeft;
		IplImage* gray_fr1 = cvCreateImage( cvGetSize(frame1), 8, 1 );
		IplImage *frame2 = grabb->imageRight;
		IplImage* gray_fr2 = cvCreateImage( cvGetSize(frame2), 8, 1 );
		imageSize = cvGetSize(frame1);
		
	
		printf("\nWant to capture %d chessboards for calibrate:", n_boards);	
		while((successes1<n_boards)||(successes2<n_boards))						
		{
			//------------- cari & drw chessboard-------------///
			if((frame++ % 20) == 0){
				//---------------- CAM KIRI-------------------------//
				result1 = cvFindChessboardCorners( frame1, board_sz,&temp1[0], &count1,CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_FILTER_QUADS);
				cvCvtColor( frame1, gray_fr1, CV_BGR2GRAY );
				//----------------CAM KANAN--------------------------------------------------------------------------------------------------------
				result2 = cvFindChessboardCorners( frame2, board_sz,&temp2[0], &count2,CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_FILTER_QUADS);
				cvCvtColor( frame2, gray_fr2, CV_BGR2GRAY );

				if(count1==n&&count2==n&&result1&&result2){
					cvFindCornerSubPix( gray_fr1, &temp1[0], count1,cvSize(11, 11), cvSize(-1,-1),cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30, 0.01) );
					cvDrawChessboardCorners( frame1, board_sz, &temp1[0], count1, result1 );
					cvShowImage( "Scan corners cam KI", frame1 );
					N1 = points[0].size();
					points[0].resize(N1 + n, cvPoint2D32f(0,0));
					copy( temp1.begin(), temp1.end(), points[0].begin() + N1 );
					++successes1;
					
					cvFindCornerSubPix( gray_fr2, &temp2[0], count2,cvSize(11, 11), cvSize(-1,-1),cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30, 0.01) );
					cvDrawChessboardCorners( frame2, board_sz, &temp2[0], count2, result2 );
					cvShowImage( "Scan corners cam KA", frame2 );
					N2 = points[1].size();
					points[1].resize(N2 + n, cvPoint2D32f(0,0));
					copy( temp2.begin(), temp2.end(), points[1].begin() + N2 );
					++successes2;
					printf("\nNumber Chessboards ditemukan: %d", successes2);
					//cvWaitKey(3000);
					//Sleep(3000);
				}else{
					cvShowImage( "corners camera2", frame2 );	
					cvShowImage( "corners camera1", frame1 );
					
				}
				grabb->stereGrabFrames();
				frame1 = grabb->imageLeft;
				cvShowImage("camera KI", frame1);
				frame2 = grabb->imageRight;
				cvShowImage("camera KA", frame2);
				
			if(cvWaitKey(1)==27) break;
			
			}
		}

		grabb->stereoGrabStopCam();
		cvDestroyWindow("camera KI");
		cvDestroyWindow("camera KA");
		cvDestroyWindow("corners camera1");
		cvDestroyWindow("corners camera2");	
		printf("\nSelesai Capture!");
		
		
		//--------------Compute for calibration-------------------
		N = n_boards*n;
		objectPoints.resize(N);
		for( i = 0; i < ny; i++ )
			for(j = 0; j < nx; j++ )   objectPoints[i*nx + j] = cvPoint3D32f(i*squareSize, j*squareSize, 0);
		for( i = 1; i < n_boards; i++ ) copy( objectPoints.begin(), objectPoints.begin() + n, objectPoints.begin() + i*n );
		npoints.resize(n_boards,n);
		
		CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
		CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
		CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
		CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
		cvSetIdentity(&_M1calib);
		cvSetIdentity(&_M2calib);
		cvZero(&_D1);
		cvZero(&_D2);
		
		printf("\nRunning stereo calibration ...");
		fflush(stdout);
		cvStereoCalibrate( &_objectPoints, &_imagePoints1, &_imagePoints2, &_npoints,&_M1calib, &_D1, &_M2calib, &_D2,imageSize, &_R, &_Tcalib, &_E, &_F,
		cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
        CV_CALIB_FIX_ASPECT_RATIO+CV_CALIB_ZERO_TANGENT_DIST + CV_CALIB_SAME_FOCAL_LENGTH );


		printf("\nDone Calibration");
		//-------------UNDISTORTION------------------------------------------
		cvUndistortPoints( &_imagePoints1, &_imagePoints1,&_M1calib, &_D1, 0, &_M1calib );
		cvUndistortPoints( &_imagePoints2, &_imagePoints2,&_M2calib, &_D2, 0, &_M2calib );
		//COMPUTE AND DISPLAY RECTIFICATION and find disparities
		CvMat* mx1calib = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
        CvMat* my1calib = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
        CvMat* mx2calib = cvCreateMat( imageSize.height,imageSize.width, CV_32F );
        CvMat* my2calib = cvCreateMat( imageSize.height,imageSize.width, CV_32F );

        double R1[3][3], R2[3][3], P1[3][4], P2[3][4];
        CvMat _R1 = cvMat(3, 3, CV_64F, R1);
        CvMat _R2 = cvMat(3, 3, CV_64F, R2);
	
            CvMat _P1 = cvMat(3, 4, CV_64F, P1);
            CvMat _P2 = cvMat(3, 4, CV_64F, P2);
			//compute variables needed for rectification using camera matrices, distortion vectors, rotation matrix, and translation vector
            cvStereoRectify( &_M1calib, &_M2calib, &_D1, &_D2, imageSize,&_R, &_Tcalib,&_R1, &_R2, &_P1, &_P2, &_Qcalib,0/*CV_CALIB_ZERO_DISPARITY*/ );
			//Precompute maps for cvRemap()
            cvInitUndistortRectifyMap(&_M1calib,&_D1,&_R1,&_P1,mx1calib,my1calib);
            cvInitUndistortRectifyMap(&_M2calib,&_D2,&_R2,&_P2,mx2calib,my2calib);
		
			

			printf("\nSaving matries for later use ...\n");
			cvSave("CalibFile//M1.yml",&_M1calib);
			cvSave("CalibFile//D1.yml",&_D1);
			cvSave("CalibFile//R1.yml",&_R1);
			cvSave("CalibFile//P1.yml",&_P1);
			cvSave("CalibFile//M2.yml",&_M2calib);
			cvSave("CalibFile//D2.yml",&_D2);
			cvSave("CalibFile//R2.yml",&_R2);
			cvSave("CalibFile//P2.yml",&_P2);
			cvSave("CalibFile//Q.yml",&_Qcalib);
			cvSave("CalibFile//T.yml",&_Tcalib);
			cvSave("CalibFile//mx1.yml",mx1calib);
			cvSave("CalibFile//my1.yml",my1calib);
			cvSave("CalibFile//mx2.yml",mx2calib);
			cvSave("CalibFile//my2.yml",my2calib);
}

void StereoFunction::stereoCorrelation(StereoGrab* grabb){
	
	int SADWindowSize = 0;
	StereoSGBM sgbm;
	int cn = cvarrToMat(img1r).channels();
	
	sgbm.preFilterCap = stereoPreFilterCap;//63; //stereoPreFilterSize;
	sgbm.SADWindowSize = stereoDispWindowSize;//3; //stereoDispWindowSize;
	sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity =10;// -39; //0
	sgbm.numberOfDisparities =  stereoNumDisparities; //144;
	sgbm.uniquenessRatio =  stereoDispUniquenessRatio; //10;
	sgbm.speckleWindowSize = 200; //200
	sgbm.speckleRange = 32;		//32
	sgbm.disp12MaxDiff = 1;		//2

	cvSplit(grabb->imageLeft,r_detect,g_detect,b_detect, NULL); 
	cvRemap( r_detect, r_detect_r, mx1, my1 ); // Undistort image
	cvRemap( g_detect, g_detect_r, mx1, my1 ); // Undistort image
	cvRemap( b_detect, b_detect_r, mx1, my1 ); // Undistort image
	cvMerge( r_detect_r, g_detect_r, b_detect_r, NULL, img_detect);


	IplImage* eq_gray = cvCreateImage(cvGetSize(img1), 8, 1);

	CvHistogram *hist;
	int hist_size = 256;
	float range[] = { 0, 256 };
	float* ranges[] = { range };

	float max_value = 0.0;
	float w_scale = 0.0;

	/* Convert the image to gray */
	cvCvtColor(grabb->imageLeft, img1, CV_RGB2GRAY);
	cvCvtColor(grabb->imageRight, img2, CV_RGB2GRAY);

	//rectification
	cvRemap( img1, img1r, mx1, my1);
	cvRemap( img2, img2r, mx2, my2);
	sgbm(cvarrToMat(img1r), cvarrToMat(img2r), cvarrToMat(disp));
	cvNormalize( disp, vdisp, 0, 256, CV_MINMAX );

	//view data
		cvNamedWindow( "Rectified", 1);
		//cvNamedWindow( "Disparity Map",1 );
		//membuat line
		CvMat part;
		cvGetCols( pair, &part, 0, imageSize.width );
		cvCvtColor( img1r, &part, CV_GRAY2BGR );
		cvGetCols( pair, &part, imageSize.width, imageSize.width*2 );
		cvCvtColor( img2r, &part, CV_GRAY2BGR );
		for( int j = 0; j < imageSize.height; j += 16 )
		cvLine( pair, cvPoint(0,j), cvPoint(imageSize.width*2,j),CV_RGB(0,255,0));
		//ending line

		//cvLine(vdisp, cvPoint(0, 120), cvPoint(320,120), CV_RGB(255, 0, 0)); //horizontal
		//cvLine(vdisp, cvPoint(160, 0), cvPoint(160, 240), CV_RGB(255, 0, 0));// vertical

		Mat dst = Mat(vdisp, true);
		//imshow("tes", dst);
		cvShowImage("Rectified", pair);
		cvShowImage("Disparity Map", vdisp);
		

		//jarak
		stereoSavePointCloud();
}

void PointCenter(cv::Point center) {
	//cout << "Center: " << center << endl;
	p_center = center;
}

void StereoFunction::stereoSavePointCloud()
{
	Mat zeroMat = zeroMat.zeros(450, 320, CV_8UC1);
		//0: fx(pixel), 1: fy(pixel), 2: base line (pixel), 3: f(mm), 4: sensor element size, 5: baseline (mm)
	double	focal = 4.2; //mm reprojectionVars[0]; //4.2126730429010615
	double	baseline = 30;	
	double 	depth = 0; 

	double vk[10] = { 2.420168, 3.638989, 4.881356, 5.843478, 6.631579, 7.730909, 7.773846, 8.365145, 8.470588, 9.960000 };
	double jk[10]= {50, 100, 150, 200, 250, 300, 350, 400, 450, 500};
	double j1=0, j2=0;
	double v1=0, v2=0;
	double jarak;

	real_disparity= cvCreateImage( imageSize, IPL_DEPTH_32F, 1 );
	cvConvertScale( disp, real_disparity, 1.0/16, 0 );
	// min 1.27 29 cm
	// max 8.16 400 cm
	if (p_center.x >= 320) p_center.x = 319;
	if (p_center.y >= 240) p_center.y = 239;
	if (p_center.x <= 0) p_center.x = 1;
	if (p_center.y <= 0) p_center.y = 1;
	//cout << "(" << p_center.x << " x " << p_center.y << ") => ";
	
	depth = (double)((baseline*focal)/((double)(cvGet2D(disp, p_center.y, p_center.x).val[0]/16))); // (y,x)
	//printf("%f\n",depth);
	
	//vk depth kalibrasi
	//jk cm
	int ok=1;
	for(int i=0; i<10; i++){
		if(depth>=vk[i] && depth<vk[i+1]){
			v1 = vk[i+1]-vk[i]; //ZCn+1 -ZCn depth kalib
			j1 = jk[i+1]-jk[i];	//ZCn+1 -ZCn cm kalib

			v2=depth-vk[i];  //Z-ZRn
			j2 = j1*v2/v1;		//j1 cm 
			j2 = j2 + jk[i];   //konversi cm
			ok=1;
			break;
		}
	}
	if(ok==1){
		//printf("%f__jarak: %f => ",depth,j2);
		//cout << "depth: " << depth << " => " << "jarak:  " << j2 << " => ";
		float phyta_dist = 0.0;
		phyta_dist = sqrt(pow(j2, 2) - pow(60, 2));
		meas_dist = phyta_dist;
		//cout << "j2: " << j2 << " phyta: " << phyta_dist << " ";
	}
	
}

void meas_distance(vector<float>& distance) {
	distance.push_back(meas_dist);
}