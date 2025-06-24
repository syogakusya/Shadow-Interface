#include "ofApp.h"
#include <algorithm>

//--------------------------------------------------------------
void ofApp::setup()
{
  cam.setup(640, 480);
  thresh = 40;
  pinchDist = 40;
  bgCaptured = false;
  colorImg.allocate(cam.getWidth(), cam.getHeight());
  grayImg.allocate(cam.getWidth(), cam.getHeight());
  bgGray.allocate(cam.getWidth(), cam.getHeight());
  diffImg.allocate(cam.getWidth(), cam.getHeight());
  threshParam.set("Thresh", thresh, 1, 255);
  pinchParam.set("PinchDist", pinchDist, 1, 200);
  gui.setup();
  gui.add(threshParam);
  gui.add(pinchParam);
  rect.setFromCenter(200, 200, 60, 60);
  dragging = false;
  pinchActive = false;
}

//--------------------------------------------------------------
void ofApp::update()
{
  cam.update();
  if (!cam.isFrameNew())
    return;
  colorImg.setFromPixels(cam.getPixels());
  grayImg = colorImg;
  thresh = threshParam;
  pinchDist = pinchParam;
  if (bgCaptured)
  {
    cv::Mat g = ofxCv::toCv(grayImg);
    cv::Mat bg = ofxCv::toCv(bgGray);
    cv::Mat diff;
    cv::absdiff(g, bg, diff);
    cv::threshold(diff, diff, thresh, 255, cv::THRESH_BINARY);
    cv::morphologyEx(diff, diff, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 2);
    diffImg.setFromPixels(diff.data, diff.cols, diff.rows);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (!contours.empty())
    {
      auto it = std::max_element(contours.begin(), contours.end(), [](const auto &a, const auto &b)
                                 { return cv::contourArea(a) < cv::contourArea(b); });
      if (cv::contourArea(*it) > 3000)
      {
        auto res = detectPinch(*it);
        gesture = res.first;
        center = res.second;
        if (gesture == "pinch")
        {
          if (!pinchActive)
          {
            if (rect.inside(center))
              dragging = true;
            pinchActive = true;
          }
          if (dragging)
            rect.setFromCenter(center, rect.getWidth(), rect.getHeight());
        }
        else
        {
          if (pinchActive)
          {
            dragging = false;
            pinchActive = false;
          }
        }
      }
    }
  }
}

//--------------------------------------------------------------
void ofApp::draw()
{
  ofSetColor(255);
  colorImg.draw(0, 0);
  ofSetColor(255, 0, 0);
  ofDrawRectangle(rect);
  if (gesture == "pinch")
  {
    ofSetColor(255, 255, 0);
    ofDrawCircle(center, 8);
  }
  gui.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
  if (key == 'b')
  {
    bgGray = grayImg;
    bgCaptured = true;
  }
  else if (key == 'r')
  {
    bgCaptured = false;
    dragging = false;
    pinchActive = false;
  }
  else if (key == OF_KEY_ESC)
  {
    ofExit();
  }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button)
{
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button)
{
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button)
{
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h)
{
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg)
{
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo)
{
}

std::pair<std::string, ofPoint> ofApp::detectPinch(const std::vector<cv::Point> &contour)
{
  cv::Moments M = cv::moments(contour);
  int cx = 0, cy = 0;
  if (M.m00 != 0)
  {
    cx = int(M.m10 / M.m00);
    cy = int(M.m01 / M.m00);
  }
  std::vector<int> hullIdx;
  cv::convexHull(contour, hullIdx, false, false);
  if (hullIdx.size() < 3)
    return {"fist", {cx, cy}};
  std::sort(hullIdx.begin(), hullIdx.end());
  std::vector<cv::Vec4i> defects;
  try
  {
    cv::convexityDefects(contour, hullIdx, defects);
  }
  catch (...)
  {
    return {"fist", {cx, cy}};
  }
  if (defects.empty())
    return {"fist", {cx, cy}};
  std::vector<cv::Point> tips;
  for (auto &d : defects)
  {
    if (d[3] / 256 < 10)
      continue;
    tips.push_back(contour[d[0]]);
    tips.push_back(contour[d[1]]);
  }
  std::vector<cv::Point> uniqueTips;
  for (auto &p : tips)
  {
    if (std::find_if(uniqueTips.begin(), uniqueTips.end(), [&](const auto &q)
                     { return p == q; }) == uniqueTips.end())
      uniqueTips.push_back(p);
  }
  if (uniqueTips.size() >= 2)
  {
    cv::Point p1 = uniqueTips[0];
    cv::Point p2 = uniqueTips[1];
    double dist = cv::norm(p1 - p2);
    ofPoint c((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
    if (dist < pinchDist)
      return {"pinch", c};
    else
      return {"open", c};
  }
  return {"fist", {cx, cy}};
}
