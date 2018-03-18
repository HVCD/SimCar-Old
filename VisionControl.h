/*****************************
Name: VisionControl
Copyright(c): ISI Lab
Author: XU Chengguang
Date: 2016/5/26
Abstract: The head file of Vision
          Control class. Detect 
		  the red mark in SigVerse
		  road environment.
******************************/
#ifndef VISION_CONTROL_H
#define VISION_CONTROL_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

using namespace std;
	
int state = 7; //current detect state
int state_last = 7; //last detect state
int stateCount = 0; //state countor

class VisionControl
{
private:
	vector<string> imgNames; //names of training images
	int trainImgNum; //number of training images
	int featureSize; //size of the HOG feature vector
	cv::Mat dataMat; //svm data mat
	cv::Mat labelMat; //svm label mat
	char* svmPath; //svm file path
	cv::Vec3b targetColor; //target detect color
	
	//image preproccess function
	cv::Mat imgPreproccess( IplImage* input );
	//feature extraction function
	vector<float> extractFeature( cv::Mat input );
public:
	//constructor
	VisionControl( int num, int fs, char* sp );
	~VisionControl();
	//train SVM classifier function
	void trainSVM();
	//state detecting function
	int detect( IplImage* input );
	//turning control function
	bool control( IplImage* input);
};

//constructor
VisionControl::VisionControl( int num, int fs, char* sp ): 
	trainImgNum(60), featureSize(324), svmPath("SVM_DATA.xml") //default value
{
	trainImgNum = num; //number of the training images
	featureSize = fs; //size of the HOG feature
	svmPath = sp; //path of the svm file
	targetColor = cv::Vec3b(0, 0, 255); //set the target color red
	
	dataMat = cv::Mat( trainImgNum, featureSize, CV_32FC1); //distribute memory for dataMat
	labelMat = cv::Mat( trainImgNum,          1, CV_32FC1); //distribute memory for labelMat
	
	/*//load names of the training images
	stringstream ss;
	int count = 0;
	while( count < trainImgNum )
	{
		ss << "E:\\train_set\\" << (count+1) << ".png";
		imgNames.push_back( ss.str() );
		ss.str("");
		ss.clear();
		count++;
	}*/
}
VisionControl::~VisionControl(){}

//image preprocces funtion
cv::Mat VisionControl::imgPreproccess( IplImage* input )
{
	cv::Mat img = cv::Mat( input ); //convert to from IplImage* to Mat
	assert( !img.empty() );
	
	cv::absdiff(
			img,
			cv::Scalar( targetColor ),
			img
		); //transform the target color to zero
	
	vector<cv::Mat> channels(3);
	cv::split(
			img,
			channels
		); //extract the channels of the picture
	
	img = channels[0] + channels[1] + channels[2]; //emphasize the target color
	cv::threshold(
			img,
			img,
			100,
			255,
			cv::THRESH_BINARY_INV | cv::THRESH_OTSU
		);// transform the traget color white and others black
		
	cv::Mat bwImg = img.clone(); //store the binary image
	
	vector<vector<cv::Point> > contours;
	cv::findContours(
			bwImg,
			contours,
			CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_NONE
		); //find the contour of the target color reign
	
	cv::Rect r;
    if( !contours.size() ) 
    {
    	cv::resize(
				img,
				img,
				cv::Size(100, 100)
			);//resize the picture
	    return img;

    }

	r = cv::boundingRect( contours[0] ); //find the outer rectangle of the ROI
	
	img = cv::Mat( img, r ); //cut the ROI
	cv::resize(
			img,
			img,
			cv::Size(100, 100)
		);//resize the picture
		
	return img; //return the preproccess result
}

//extract the feature function
vector<float> VisionControl::extractFeature( cv::Mat input )
{
	vector<float> descriptor; //create  the HOG descriptor

	for( int i = 0; i < 100; i++ )
	{
		for( int j = 0; j < 100; j++ )
		{
			if( input.at<uchar>(i,j) )
				input.at<uchar>(i,j) = 255;
			else
				input.at<uchar>(i,j) = 0;
		}
	}//normalization the preprocess images
	
	cv::HOGDescriptor *hog = new cv::HOGDescriptor(cv::Size(100,100),cv::Size(50,50),cv::Size(25,25),cv::Size(25,25),9); //create the HOG pointer
	hog->compute(
			input,
			descriptor,
			cv::Size(1,1),
			cv::Size(0,0)
		);//extract HOG feature
		
	hog = NULL; //release the pointer
	return descriptor;
}

//training the SVM classifier
void VisionControl::trainSVM()
{
	IplImage* input = NULL;
	char* path; //store image path
	cv::Mat img; //store temp image
 	vector<float> tmp; //temp feature vector

	for(int i = 0; i < imgNames.size(); i++ )
	{
		path = const_cast<char*>(imgNames[i].c_str()); //load the path of each image
		input = cvLoadImage( path ); //load image
		assert( input ); //check the loading result

		img = imgPreproccess( input ); //preproccess image
		tmp = extractFeature( img ); //extract feature

		for( int j = 0; j < featureSize; j++ )
		{
			dataMat.at<float>(i, j) = tmp[j];
		} //load the feature to the data mat
		
		if( i < 4 ) //load the label mat
			labelMat.at<float>(i,0) = 1;

		if( i >= 4 && i < 7 )
			labelMat.at<float>(i,0) = 2;

		if( i >= 7 && i < 11 )
			labelMat.at<float>(i,0) = 3;

		if( i >= 11 && i < 15  )
			labelMat.at<float>(i,0) = 4;

		if( i >= 15 && i < 19 )
			labelMat.at<float>(i,0) = 5;

		if( i >= 19 && i < 23 )
			labelMat.at<float>(i,0) = 6;
	}// load the data into data mat and label mat

	//SVM classifier
	CvSVM svm;  //create a svm clasifier
	CvSVMParams params; //parameters
	CvTermCriteria criteria; //ending judgements
	//set params
	criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );        
    params = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria );   
	//svm training
	svm.train(
			dataMat,
			labelMat,
			cv::Mat(),
			cv::Mat(),
			params
		);

	svm.save( svmPath ); //save the svm result in target file
}

//detection
int VisionControl::detect( IplImage* input )
{
	cv::Mat img; //img Mat
	float result; //classify result
	vector<float> feature; //temp feature
 	CvSVM svm; //create svm object

	svm.load( svmPath ); //load svm file

	img = imgPreproccess( input ); //preproccess
	//imshow("img", img);
	feature = extractFeature( img ); //extract feature

	cv::Mat sample(1, feature.size(), CV_32FC1); //convert to svm mat
	for( int i = 0; i < feature.size(); i++ )
		sample.at<float>(0,i) = feature[i];

    result = svm.predict( sample ); //svm classify
	
	if( result == state_last ) //judge the current state by accumulating algorithm
	{
		stateCount++;
		if( stateCount >= 16 ) //accumulate up to 15 is a good state
		{
			state = result;
			stateCount = 0;
		}
	}else
	{
		stateCount = 0;
	}
	
	state_last = result; 
	return state; //return the good state
}

bool VisionControl::control( IplImage* input )
{
	bool turn = false; //turnig control flag
	cv::Mat img = cv::Mat( input );
	assert( !img.empty() );

	cv::absdiff(
			img,
			cv::Scalar( targetColor ),
			img
		); //transfrom the target color to zero
	
	vector<cv::Mat> channels(3);
	cv::split(
			img,
			channels
		);
	
	img = channels[0] + channels[1] + channels[2]; //emphasize the target color
	cv::threshold(
			img,
			img,
			120,
			255,
			cv::THRESH_BINARY_INV
		);
	
	//cv::imshow("Mark", img);
	cv::waitKey(1);
	
	cv::Mat bwImg = img.clone(); //store the binary image
	
	//cut the ROI
	vector<vector<cv::Point> > contours;
	cv::findContours(
			bwImg,
			contours,
			CV_RETR_EXTERNAL,
			CV_CHAIN_APPROX_NONE
		); //find the contour of the ROI

	if( contours.size() == 0 ) 
	{
		turn = true;

	} //determine when to turn
		
	return turn;
}

#endif
