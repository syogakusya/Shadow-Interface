#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"
#include "ofxGui.h"
#include "GestureDetector.h"

class ofApp : public ofBaseApp
{

public:
	void setup();
	void update();
	void draw();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);

	ofVideoGrabber cam;
	ofxCvColorImage colorImg;
	ofxCvGrayscaleImage grayImg;
	ofxCvGrayscaleImage bgGray;
	ofxCvGrayscaleImage diffImg;
	bool bgCaptured;
	int thresh;
	int pinchDist;
	int minPinchDist;
	int minArea;
	int maxArea;
	int minDistance;
	int shadowMargin;
	int maxContours;

	ofxPanel gui;
	ofParameter<bool> autoThreshParam;
	ofParameter<int> threshParam;
	ofParameter<int> pinchParam;
	ofParameter<int> minPinchParam;
	ofParameter<int> minAreaParam;
	ofParameter<int> maxAreaParam;
	ofParameter<int> minDistanceParam;
	ofParameter<int> shadowMarginParam;
	ofParameter<int> maxContoursParam;
	ofxLabel shadowHandStatus;

	ofRectangle rect;
	bool dragging;
	bool pinchActive;

	std::vector<std::string> gestures;
	std::vector<ofPoint> centers;

	std::vector<ofPoint> quadPoints;
	bool isCalibrating;
	cv::Mat perspective;
	ofxCvColorImage warpedImg;

	GestureDetector gestureDetector;
};
