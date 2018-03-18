#include "ControllerEvent.h"  
#include "Controller.h"  
#include "Logger.h"  

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "VisionControl.h"

#define	PI	3.14159265357978624
#define DEG2RAD(DEG) ( (PI) * (DEG) / 180.0 )

const double WheelDistance = 2.0;
const double WheelRadius = 2.0;
const double TurnRadius = 2.0;
const double NormalVelcoity = 2.0;

const int road = 7;
int COUNT = 1;  
class MyController : public Controller 
{  
public:  
	void onInit(InitEvent &evt);  
	double onAction(ActionEvent&);  
	void onRecvMsg(RecvMsgEvent &evt); 
	void onCollision(CollisionEvent &evt); 
	
public:
	int onFuseEEG_Image(int v_EEG, int v_Image);
	bool onWaitOrder( int v_Image );
	void onMoveCar(int direction, int v_Image, bool flag);
public:
	RobotObj *m_robot;
	ViewService *view;
	ViewImage *img;
	IplImage *image;
	
	//int m_h_com;
	//int m_l_com;
	int m_time;
	bool b_wait;
	
	int i_msg;	// command from EEG
	int fuse;	// fuse command
	
	ofstream output;
};  
  
void MyController::onInit(InitEvent &evt) 
{  
	m_robot = getRobotObj(myname());
	m_robot->setWheel(WheelRadius, WheelDistance);
	
	view = (ViewService*)connectToService("SIGViewer");
	image = cvCreateImage(cvSize(320, 240), 8, 3);
	//m_h_com = 0;
	//m_l_com = 0;
	m_time = 0;
	b_wait = false;
	
	i_msg = 0;
	fuse = 0;
	
	output.open("data.txt");
}  

int MyController::onFuseEEG_Image(int v_EEG, int v_Image)
{
	int direction = 0;	//0 straight, 1 left, 2 right
	switch(v_Image)
	{
		case 1:	// cross
			fuse = 7;	// 00111
			direction = rand() % 3;
			break;
		case 2:	// right corner
			fuse = 9;	// 01001
			direction = 2;
			break;
		case 3:	// left corner
			fuse = 10;	// 01010
			direction = 1;
			break;
		case 4:	// left T
			fuse = 6;	// 00110
			direction = rand() % 2;
			break;
		case 5:	// right T
			fuse = 5;	// 00101
			direction = (rand() % 2) * 2;
			break;
		case 6:	// T
			fuse = 3;	// 00011
			direction = (rand() % 2) + 1;
			break;
		default:
			fuse = 4;	// 00100
			direction = 0;
	}
	int fuse_result = fuse & i_msg;
	std::cout << "vImage = " << v_Image << "\t" << v_EEG << "\t" << fuse_result << "\t";
	output << v_Image << "\t" << v_EEG << "\t" << fuse_result << "\t";
		
	if((fuse_result & 8) == 8 && (fuse == 9 || fuse == 10))	// 01000 stop
	{
		direction = 3;
	}
	else if(fuse_result > 0)
	{
		switch(fuse_result & 7)	// 111
		{
			case 4:	// 100 straight
				direction = 0;
				break;
			case 2:	// 010 left
				direction = 1;
				break;
			case 1:	// 001 right
				direction = 2;
				break;
			default:
				std::cerr << "fuse_result case error" << std::endl;
				//exit(1);
		}
	}
	//std::cout << "Direction = " << direction << std::endl;
	return direction;
}

bool MyController::onWaitOrder( int v_Image )
{
	
	if( v_Image == 7 )
		return false;

    std::stringstream ss;
    
	while( true )
	{
		img = view->captureView(4, COLORBIT_24, IMAGE_320X240);
		memcpy(image->imageData, img->getBuffer(), image->imageSize); //load image
		cv::Mat img = cv::Mat( image );
		assert( !img.empty() );
		
		if( COUNT % 15 == 0 )
		{
			ss << "/home/robocup/image/target/" << COUNT << ".jpg";
    		cv::imwrite( ss.str(), img);
    		ss.str("");
    	}
    	COUNT++;
    	
		cv::absdiff(
				img,
				cv::Scalar( cv::Vec3b(0, 0, 255) ),
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
			return true;
		} //determine when to turn
	}
}

void MyController::onMoveCar(int direction, int v_Image, bool flag)
{
	std::cout << direction << std::endl;
	output << direction << std::endl;
	if(v_Image == 7 || flag == false)
		return;
		 
	if( (v_Image == 2) || (v_Image == 3) || (v_Image == 6)  ) //sleep some time when T & corner 
	  sleep(1.2);
			
	switch(direction) //steering control
	{
	case 1:
			m_robot->setWheelVelocity(TurnRadius * PI / 4.0, (TurnRadius + WheelDistance) * PI / 4.0 + 0.0015);
			sleep(1.0);
			m_robot->setWheelVelocity(NormalVelcoity, NormalVelcoity);
			break;
	case 2:
			m_robot->setWheelVelocity((TurnRadius + WheelDistance) * PI / 4.0 + 0.0015, TurnRadius * PI / 4.0);
			sleep(1.0);
			m_robot->setWheelVelocity(NormalVelcoity, NormalVelcoity);
			break;
	case 3:
			m_robot->setWheelVelocity(0.0, 0.0);
			break;
	}
	direction = 0; //reset
	i_msg = 0; //reset
	state = state_last = 7; //reset
}
	
double MyController::onAction(ActionEvent &evt) 
{ 
    
	img = view->captureView(4, COLORBIT_24, IMAGE_320X240); 	   
    memcpy(image->imageData, img->getBuffer(), image->imageSize);
    
    VisionControl m_vc(23, 324, "svm.xml");
   
    int result = m_vc.detect(image);	// 判断路口类型
    std::cout << "Result = " << result << std::endl;
    int fuse_result = onFuseEEG_Image(i_msg, result); // 融合
    bool order = onWaitOrder( result );
    onMoveCar( fuse_result, result, order );// 转向判断执行
	return 0.1;  
}
void MyController::onRecvMsg(RecvMsgEvent &evt) 
{  
	static bool m_begin = false;
	std::string sender = evt.getSender();

	char *all_msg = (char*)evt.getMsg();
	std::string msg;
	msg = evt.getMsg();
	
	if(msg == "g" && !m_begin)
	{
		m_robot->setWheelVelocity(NormalVelcoity, NormalVelcoity);
		m_begin = true;
	}
	else if(msg == "l")
	{
		m_robot->setWheelVelocity(TurnRadius * PI / 4.0, 
						(TurnRadius + WheelDistance) * PI / 4.0 + 0.001);
		sleep(1.0);
		m_robot->setWheelVelocity(NormalVelcoity, NormalVelcoity);
	}
	else if(msg == "r")
	{
		m_robot->setWheelVelocity((TurnRadius + WheelDistance) * PI / 4.0 + 0.001, 
						TurnRadius * PI / 4.0);
		sleep(1.0);
		m_robot->setWheelVelocity(NormalVelcoity, NormalVelcoity);
	}
	else if(msg == "s")
	{
		m_robot->setWheelVelocity(0.0, 0.0);
	}
	
	i_msg = atoi(all_msg);
	//std::cout << "\t\t\t\tmsg from brain = " << i_msg << std::endl;
	
}  

void MyController::onCollision(CollisionEvent &evt) 
{ 
}
  
extern "C" Controller * createController() 
{  
	return new MyController;  
}  

