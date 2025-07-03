#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"
#include "ofxGui.h"

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

	ofxPanel gui;
	ofParameter<int> threshParam;
	ofParameter<int> pinchParam;
	ofxLabel shadowHandStatus;

	ofRectangle rect;
	bool dragging;
	bool pinchActive;

	std::string gesture;
	ofPoint center;

	std::pair<std::string, ofPoint> detectPinch(const std::vector<cv::Point> &contour);

	std::vector<ofPoint> quadPoints;
	bool isCalibrating;
	cv::Mat perspective;
	ofxCvColorImage warpedImg;

	std::vector<ofPoint> contourPts;
	std::vector<ofPoint> hullPts;
	std::vector<ofPoint> tipsPts;
};
